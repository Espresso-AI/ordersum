import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from itertools import permutations
from src.rouge import RougeScorer

scorer = RougeScorer(['rougeL'])


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def _cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_oracle_algorithm(text: List[str], abstractive: List[str], summary_size:int):
    max_rouge = 0.0
    text = [_rouge_clean(' '.join([sent])).split() for sent in text]
    abstractive = _rouge_clean(' '.join(abstractive)).split()

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in text]
    reference_1grams = _get_word_ngrams(1, [abstractive])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in text]
    reference_2grams = _get_word_ngrams(2, [abstractive])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(text)):
            if i in selected:
                continue

            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            rouge_1 = _cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = _cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if cur_id == -1:
            return sorted(selected)

        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def order_oracle(text: List[str], abstractive: List[str], oracle: List[int]):
    candidates = list(permutations(oracle))
    scores = []
    for c in candidates:
        summary = [text[i] for i in c]
        rouge = scorer.score('\n'.join(abstractive), '\n'.join(summary))
        score = rouge['rougeL'].fmeasure
        scores.append(score)

    return candidates[np.argmax(scores)]


def save_ordered_extractive(dataset: pd.DataFrame, save_path: str):
    ordered = []
    for i in tqdm(range(len(dataset))):
        sample = dataset.iloc[i]
        text = sample['text']
        abstractive = sample['abstractive']
        extractive = sample['extractive']

        ordered.append({
            'text': text,
            'unordered': extractive,
            'ordered': order_oracle(text, abstractive, extractive),
        })
    with open(save_path, 'w') as f:
        json.dump(ordered, f, indent=4)
