import os
import datetime
import pandas as pd
import pytorch_lightning as pl
from typing import Tuple
from torch.optim import AdamW
from src.model.ordersum import *
from src.model.utils import select_candidate, convert_checkpoints
from src.utils.lr_scheduler import *
from src.rouge.rouge_score import RougeScorer


class OrderSum_Engine(pl.LightningModule):

    def __init__(
            self,
            model,
            train_df: Optional[pd.DataFrame] = None,
            val_df: Optional[pd.DataFrame] = None,
            test_df: Optional[pd.DataFrame] = None,
            num_can: Optional[int] = 5,
            n_block: int = 3,
            model_checkpoint: Optional[str] = None,
            freeze_base: bool = False,
            lr: float = None,
            betas: Tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 0.0,
            adam_epsilon: float = 1e-8,
            num_warmup_steps: int = None,
            num_training_steps: int = None,
            lr_init_eps: float = 0.1,
            save_result: bool = False,
    ):
        super().__init__()

        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.num_can = num_can
        self.n_block = n_block

        # hparmas
        self.model_checkpoint = model_checkpoint
        self.freeze_base = freeze_base
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_init_eps = lr_init_eps
        self.save_result = save_result

        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum', 'rougeL'])
        self.prepare_training()

    def prepare_training(self):
        self.model.train()

        if self.model_checkpoint:
            checkpoint = convert_checkpoints(self.model_checkpoint)
            self.model.load_state_dict(checkpoint)

        if self.freeze_base:
            for p in self.model.base_model.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_params, self.lr, betas=self.betas, eps=self.adam_epsilon)
        scheduler = get_transformer_scheduler(optimizer, self.num_warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {'scheduler': scheduler, 'interval': 'step'}
        }

    def training_step(self, batch, batch_idx):
        texts, ref_sums = [], []
        for i in batch['id']:
            sample = self.train_df[self.train_df['id'] == i].squeeze()
            text = sample['text']
            texts.append(text)

            ref_sum = sample['abstractive']
            ref_sums.append('\n'.join(ref_sum))

        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
            texts,
            ref_sums
        )
        loss, sent_loss, sum_loss \
            = outputs['loss'], outputs['sentence_loss'], outputs['summary_loss']

        self.log('train_step_loss', loss, prog_bar=True)
        self.log('train_step_sent_loss', sent_loss, prog_bar=True)
        self.log('train_step_sum_loss', sum_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
        )
        loss, sent_loss, sum_loss \
            = outputs['loss'], outputs['sentence_loss'], outputs['summary_loss']
        preds = outputs['prediction']

        ref_sums, can_sums = [], []
        for i, id in enumerate(batch['id']):
            sample = self.val_df[self.val_df['id'] == id].squeeze()
            text = sample['text']

            ref_sum = sample['abstractive']
            ref_sums.append('\n'.join(ref_sum))

            if self.n_block:
                can_sum, _ = select_candidate(text, preds[i], self.num_can, self.n_block)
            else:
                pred = preds[i][:self.sum_size]
                can_sum = [text[p] for p in pred]

            can_sums.append('\n'.join(can_sum))

        return loss, sent_loss, sum_loss, ref_sums, can_sums

    def validation_epoch_end(self, val_steps):
        losses, sent_losses, sum_losses = [], [], []
        r1, r2, rL = [], [], []

        print('calculating ROUGE score...')
        for loss, sent_loss, sum_loss, ref_sums, can_sums in val_steps:
            for ref_sum, can_sum in zip(ref_sums, can_sums):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

            losses.append(loss)
            sent_losses.append(sent_loss)
            sum_losses.append(sum_loss)

        n = len(losses)
        loss = sum(losses) / n
        sent_loss = sum(sent_losses) / n
        sum_loss = sum(sum_losses) / n

        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_sent_loss', sent_loss, prog_bar=True)
        self.log('val_sum_loss', sum_loss, prog_bar=True)
        self.log('val_rouge1', r1, prog_bar=True)
        self.log('val_rouge2', r2, prog_bar=True)
        self.log('val_rougeL', rL, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
        )
        preds = outputs['prediction']

        texts, ref_sums, can_sums, can_preds = [], [], [], []
        for i, id in enumerate(batch['id']):
            sample = self.test_df[self.test_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            ref_sum = sample['abstractive']
            ref_sums.append('\n'.join(ref_sum))

            if self.n_block:
                can_sum, pred = select_candidate(text, preds[i], self.num_can, self.n_block)
            else:
                pred = preds[i][0]
                can_sum = [text[p] for p in pred]

            can_sums.append('\n'.join(can_sum))
            can_preds.append(pred)

        return texts, ref_sums, can_sums, can_preds

    def test_epoch_end(self, test_steps):
        result = {
            'text': [],
            'reference summary': [],
            'candidate summary': [],
            'prediction': [],
        }
        r1, r2, rLsum, rL = [], [], [], []

        print('calculating ROUGE score...')
        for texts, ref_sums, can_sums, can_preds in test_steps:
            for i, (ref_sum, can_sum) in enumerate(zip(ref_sums, can_sums)):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rLsum.append(rouge['rougeLsum'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

                if self.save_result:
                    result['text'].append(texts[i])
                    result['reference summary'].append(ref_sum)
                    result['candidate summary'].append(can_sum)
                    result['prediction'].append(can_preds[i])

        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rLsum = 100 * (sum(rLsum) / len(rLsum))
        rL = 100 * (sum(rL) / len(rL))

        print('rouge1: ', r1)
        print('rouge2: ', r2)
        print('rougeLsum: ', rLsum)
        print('rougeL: ', rL)

        if self.save_result:
            path = 'result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)
