import torch
import torch.nn as nn
import random
from typing import Optional, List
from itertools import permutations, combinations
from transformers import BartForConditionalGeneration, BatchEncoding
from src.model.extractor.encoder import SumEncoder
from src.rouge import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.Tensor

scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])


class OrderSum_BART(nn.Module):

    def __init__(
            self,
            bart_tokenizer,
            base_checkpoint: str,
            num_ext_sent: int = 5,
            num_can_sent: List[int] = (2, 3),
            num_can_train: Optional[int] = None,
            num_can_valid: Optional[int] = None,
            enc_num_layers: int = 0,
            enc_intermediate_size: int = 2048,
            enc_num_attention_heads: int = 8,
            enc_dropout_prob: float = 0.1,
            margin: float = 0.01,
            alpha: float = 1.0,
            beta: float = 1.0,
            pool: str = 'average',
    ):
        super().__init__()

        if len(bart_tokenizer) != bart_tokenizer.vocab_size + 3:
            raise ValueError('BartTokenizer added cls, sep, doc must be given')

        self.base_checkpoint = base_checkpoint
        bart = BartForConditionalGeneration.from_pretrained(self.base_checkpoint)
        bart.resize_token_embeddings(len(bart_tokenizer))
        self.base_model = bart.get_encoder()

        enc_hidden_size = bart.config.max_position_embeddings

        self.head = SumEncoder(
            enc_num_layers,
            enc_hidden_size,
            enc_intermediate_size,
            enc_num_attention_heads,
            enc_dropout_prob,
        ).eval()

        self.num_ext_sent = num_ext_sent
        self.num_can_sent = sorted(num_can_sent)
        self.num_can_train = num_can_train
        self.num_can_valid = num_can_valid

        self.margin = margin
        self.alpha = alpha
        self.beta = beta

        self.sentence_loss = nn.BCELoss(reduction='none')

        if pool == 'average':
            self.pooling = nn.AdaptiveAvgPool1d(enc_hidden_size)
        elif pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(enc_hidden_size)
        else:
            raise ValueError("Pooling must be one of 'average', 'mean'")

    def forward(
            self,
            encodings: BatchEncoding,
            cls_token_ids: Tensor,
            ext_labels: Optional[Tensor] = None,
            texts: List[List[str]] = None,
            references: List[str] = None,
    ):
        token_embeds = self.base_model(**encodings).last_hidden_state
        doc_embeds = token_embeds[:, 0, :]
        cls_embeds, cls_mask, cls_logits = self.head(token_embeds, cls_token_ids).values()

        # sentence-level
        sent_scores = torch.sigmoid(cls_logits) * cls_mask

        # summary-level
        batch = doc_embeds.size(0)
        num_sents = torch.sum(cls_mask, dim=-1)
        sim_scores, can_sum_ids, can_embeds = [], [], []

        for i in range(batch):
            text = texts[i] if texts else None
            reference = references[i] if references else None

            candidate = self.match(
                doc_embeds[i],
                cls_embeds[i],
                sent_scores[i],
                num_sents[i],
                text,
                reference
            )
            sim_scores.append(candidate['similarity'])
            can_sum_ids.append(candidate['candidate_ids'])
            can_embeds.append(candidate['candidate_embeddings'])

        # calculate loss
        if ext_labels is not None:
            sent_loss = self.sentence_loss(sent_scores, ext_labels.float())
            sent_loss = (sent_loss * cls_mask).sum() / num_sents.sum()

            sum_loss = 0.0
            for sim_score in sim_scores:
                sum_loss = sum_loss + self.candidate_loss(sim_score)
            sum_loss = sum_loss / batch

            total_loss = (self.alpha * sent_loss) + (self.beta * sum_loss)
        else:
            total_loss, sent_loss, sum_loss = None, None, None

        # prediction
        prediction, confidence = [], []
        for i, score in enumerate(sim_scores):
            conf, order = torch.sort(score, descending=True, dim=-1)
            pred = [can_sum_ids[i][j] for j in order]
            prediction.append(pred)
            confidence.append(conf)

        return {
            'loss': total_loss,
            'prediction': prediction,
            'confidence': confidence,
            'sentence_loss': sent_loss,
            'summary_loss': sum_loss,
        }

    def match(self, doc_embed, cls_embeds, sent_score, num_sent, text, reference):
        can_sent_ids = torch.topk(sent_score, self.num_ext_sent, dim=-1).indices
        can_sent_ids = [int(i) for i in can_sent_ids if i < num_sent]

        # make candidiates
        if len(can_sent_ids) < min(self.num_can_sent):
            can_sum_ids = list(permutations(can_sent_ids, len(can_sent_ids)))
        else:
            can_sum_ids = [list(permutations(can_sent_ids, i)) for i in self.num_can_sent]
            can_sum_ids = sum(can_sum_ids, [])

        # anchor_sampling
        sample = None
        if self.training and self.num_can_train:
            sample = self.anchor_sampling(can_sum_ids, self.num_can_train, can_sent_ids, self.num_can_sent)
        if not self.training and self.num_can_valid:
            sample = self.anchor_sampling(can_sum_ids, self.num_can_valid, can_sent_ids, self.num_can_sent)
        if sample:
            can_sum_ids = [can_sum_ids[i] for i in sample]

        # embedding the candidate summaries
        can_embeds = []
        for can_ids in can_sum_ids:
            can_embed = cls_embeds[torch.tensor(can_ids)]
            can_embed = can_embed.contiguous().view(1, -1)
            can_embed = self.pooling(can_embed)
            can_embeds.append(can_embed.squeeze())

        if self.training:
            if text and reference:
                can_sum_ids, can_embeds = self.sort_by_metric(text, reference, can_sum_ids, can_embeds)
            else:
                raise ValueError("texts and references are necessary for training OrderSum-Ext")

        can_embeds = torch.stack(can_embeds, dim=0)
        sim_score = torch.cosine_similarity(can_embeds, doc_embed, dim=-1)

        return {
            'similarity': sim_score,
            'candidate_ids': can_sum_ids,
            'candidate_embeddings': can_embeds
        }

    @classmethod
    def anchor_sampling(cls, can_sum_ids, num_can_train, can_sent_ids, num_can_sent):
        num_sample = len(can_sum_ids)

        if num_can_train and num_can_train < num_sample:
            sample = []
            # candidates selected in CoLo are basically included
            anchors = [list(combinations(can_sent_ids, i)) for i in num_can_sent]
            anchors = sum(anchors, [])

            for i, can_ids in enumerate(can_sum_ids):
                if can_ids in anchors:
                    sample.append(i)

            while len(sample) < num_can_train:
                i = random.sample(range(0, num_sample), 1)[0]
                if i not in sample:
                    sample.append(i)

            return sorted(sample)
        else:
            return None

    @classmethod
    def sort_by_metric(cls, text, reference, can_sum_ids, can_embeds):
        metrics = []
        for can_ids in can_sum_ids:
            candidate = '\n'.join([text[i] for i in can_ids])
            # sort by ROUGE score
            score = scorer.score(reference, candidate)
            score = score['rouge1'].fmeasure + score['rouge2'].fmeasure + score['rougeL'].fmeasure
            metrics.append(score)

        can_sum_ids = sorted(zip(can_sum_ids, metrics), key=lambda x: x[1], reverse=True)
        can_sum_ids = [i[0] for i in can_sum_ids]
        can_embeds = sorted(zip(can_embeds, metrics), key=lambda x: x[1], reverse=True)
        can_embeds = [i[0] for i in can_embeds]

        return can_sum_ids, can_embeds

    def candidate_loss(self, sim_score):
        loss = nn.MarginRankingLoss(margin=0.0)(
            sim_score,
            sim_score,
            torch.ones(sim_score.size()).to(device)
        )
        num_cand = sim_score.size(0)

        for i in range(1, num_cand):
            pos_score = sim_score[:-i]
            neg_score = sim_score[i:]

            loss_fn = nn.MarginRankingLoss(self.margin * i)
            loss = loss + loss_fn(
                pos_score,
                neg_score,
                torch.ones(pos_score.size()).to(device)
            )
        return loss
