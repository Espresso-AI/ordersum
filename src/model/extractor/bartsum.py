import torch
import torch.nn as nn
from typing import Optional
from transformers import BartForConditionalGeneration, BatchEncoding
from .encoder import SumEncoder

Tensor = torch.Tensor


class BartSum_Ext(nn.Module):

    def __init__(
            self,
            bart_tokenizer,
            base_checkpoint: str,
            enc_num_layers: int = 0,
            enc_intermediate_size: int = 2048,
            enc_num_attention_heads: int = 8,
            enc_dropout_prob: float = 0.1,
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

        self.sentence_loss = nn.BCELoss(reduction='none')

    def forward(
            self,
            encodings: BatchEncoding,
            cls_token_ids: Tensor,
            ext_labels: Optional[Tensor] = None,
    ):
        token_embeds = self.base_model(**encodings).last_hidden_state
        _, cls_mask, cls_logits = self.head(token_embeds, cls_token_ids).values()

        scores = torch.sigmoid(cls_logits) * cls_mask
        num_sents = torch.sum(cls_mask, dim=-1)

        loss = None
        if not (self.sentence_loss is None or ext_labels is None):
            loss = self.sentence_loss(scores, ext_labels.float())
            loss = (loss * cls_mask).sum() / num_sents.sum()

        prediction, confidence = [], []
        for i, score in enumerate(scores):
            conf, pred = torch.sort(score[cls_mask[i] == 1], descending=True, dim=-1)
            prediction.append(pred.tolist())
            confidence.append(conf)

        return {
            'logits': cls_logits,
            'loss': loss,
            'prediction': prediction,
            'confidence': confidence,
        }
