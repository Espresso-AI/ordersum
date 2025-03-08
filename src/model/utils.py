import torch
from collections import OrderedDict
from typing import List, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_checkpoints(ckpt_path):
    pl_ckpt = torch.load(ckpt_path, map_location=device)
    pl_ckpt = pl_ckpt['state_dict']

    assert next(iter(pl_ckpt)).split('.')[0] == 'model', \
        'This function is to convert pl.LightningModule checkpoints to nn.Module checkpoints of the model.'

    ckpt = {k.split('.', 1)[1]: v for k, v in pl_ckpt.items()}
    ckpt = OrderedDict(ckpt)
    return ckpt


def get_ngrams(
        tokens: List[str],
        n: int,
):
    ngrams = set()
    num = len(tokens) - n
    for i in range(num + 1):
        ngrams.add(tuple(tokens[i:i + n]))
    return ngrams


def ngram_blocking(
        sent: str,
        can_sum: List[str],
        ngram: int,
):
    sent_tri = get_ngrams(sent.split(), ngram)
    for can_sent in can_sum:
        can_tri = get_ngrams(can_sent.split(), ngram)
        if len(sent_tri.intersection(can_tri)) > 0:
            return True
    return False


def tri_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 3)


def quad_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 4)


def get_candidate_sum(
        text: List[str],
        prediction: List[int],
        sum_size: Optional[int] = None,
        n_block: int = 3
):
    last_sum, last_pred = [], []

    for i, sent_id in enumerate(prediction):
        sent = text[sent_id]
        if not ngram_blocking(sent, last_sum, n_block):
            last_sum.append(sent)
            last_pred.append(sent_id)

        if sum_size and (len(last_sum) == sum_size):
            break

    return last_sum, last_pred


def candidate_blocking(
        can_sum: List[str],
        n_block: int
):
    pre_sum = []
    for sent in can_sum:
        if not ngram_blocking(sent, pre_sum, n_block):
            pre_sum.append(sent)
        else:
            return True
    return False


def select_candidate(
        text: List[str],
        prediction: List[List],
        num_can: Optional[int] = 5,
        n_block: int = 3
):
    prediction = prediction[:num_can]
    last_sum = None
    last_pred = None

    for pred in prediction:
        can_sum = [text[i] for i in pred]
        if not candidate_blocking(can_sum, n_block):
            last_sum = can_sum
            last_pred = pred
            break

    if last_sum:
        return last_sum, last_pred
    else:
        last_pred = prediction[0]
        last_sum = [text[i] for i in last_pred]
        return last_sum, last_pred