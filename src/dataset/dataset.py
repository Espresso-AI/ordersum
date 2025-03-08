import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Bart_Dataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            bart_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = None,
            cls_token: str = "<cls>",
            sep_token: str = "<sep>",
            doc_token: str = "<doc>"
    ):
        if len(bart_tokenizer) != bart_tokenizer.vocab_size + 3:
            raise ValueError('3 tokens, cls, sep, doc must be added for BART-based models')

        self.data = data
        self.tokenizer = bart_tokenizer
        self.max_seq_len = max_seq_len
        self.pad = self.tokenizer.pad_token_id

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.doc_token = doc_token

        self.cls = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.sep = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.doc = self.tokenizer.convert_tokens_to_ids(self.doc_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # load and tokenize each sentence
        encodings = []
        for sent in row['text']:
            encoding = self.tokenizer(
                sent,
                add_special_tokens=False,
            )
            encoding['input_ids'] = [self.cls] + encoding['input_ids'] + [self.sep]
            encoding['attention_mask'] = [1] + encoding['attention_mask'] + [1]
            encodings.append(encoding)

        input_ids = [self.doc]
        token_type_ids = [0]
        attention_mask = [1]
        cls_token_ids = []
        ext_label = []

        # seperate each of sequences
        seq_id = 0
        for enc in encodings:
            if seq_id > 1:
                seq_id = 0
            cls_token_ids += [len(input_ids)]
            input_ids += enc['input_ids']
            token_type_ids += len(enc['input_ids']) * [seq_id]
            attention_mask += len(enc['input_ids']) * [1]

            if encodings.index(enc) in row['extractive']:
                ext_label += [1]
            else:
                ext_label += [0]

            seq_id += 1

        # pad and truncate inputs
            if len(input_ids) == self.max_seq_len:
                break

            elif len(input_ids) > self.max_seq_len:
                sep = input_ids[-1]
                input_ids = input_ids[:self.max_seq_len - 1] + [sep]
                token_type_ids = token_type_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]
                break

        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += pad_len * [self.pad]
            token_type_ids += pad_len * [0]
            attention_mask += pad_len * [0]

        # adjust for BartSum_Ext
        if len(cls_token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(cls_token_ids)
            cls_token_ids += pad_len * [-1]
            ext_label += pad_len * [0]

        # BART encodings only need 'input_ids' and 'attention_mask'
        encodings = BatchEncoding(
            {
                'input_ids': torch.tensor(input_ids).long().to(device),
                'attention_mask': torch.tensor(attention_mask).to(device),
            }
        )
        return dict(
            id=row['id'],
            encodings=encodings,
            cls_token_ids=torch.tensor(cls_token_ids).to(device),
            ext_label=torch.tensor(ext_label).to(device)
        )


class Bert_Dataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = None,
            doc_token: str = "[DOC]",
    ):
        if not tokenizer.vocab["[DOC]"] == tokenizer.vocab_size:
            raise ValueError('[DOC] must be added, and the only added token for CoLo_Dataset')

        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad = tokenizer.pad_token_id

        self.doc_token = doc_token
        self.doc = self.tokenizer.convert_tokens_to_ids(self.doc_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # load and tokenize each sentence
        encodings = []
        for sent in row['text']:
            encoding = self.tokenizer(
                sent,
                add_special_tokens=True,
            )
            encodings.append(encoding)

        input_ids = [self.doc]
        token_type_ids = [0]
        attention_mask = [1]
        cls_token_ids = []
        ext_label = []

        # seperate each of sequences
        seq_id = 0
        for enc in encodings:
            if seq_id > 1:
                seq_id = 0
            cls_token_ids += [len(input_ids)]
            input_ids += enc['input_ids']
            token_type_ids += len(enc['input_ids']) * [seq_id]
            attention_mask += len(enc['input_ids']) * [1]

            if encodings.index(enc) in row['extractive']:
                ext_label += [1]
            else:
                ext_label += [0]

            seq_id += 1

        # pad and truncate inputs
            if len(input_ids) == self.max_seq_len:
                break

            elif len(input_ids) > self.max_seq_len:
                sep = input_ids[-1]
                input_ids = input_ids[:self.max_seq_len - 1] + [sep]
                token_type_ids = token_type_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]
                break

        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += pad_len * [self.pad]
            token_type_ids += pad_len * [0]
            attention_mask += pad_len * [0]

        # adjust for BertSum_Ext
        if len(cls_token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(cls_token_ids)
            cls_token_ids += pad_len * [-1]
            ext_label += pad_len * [0]

        encodings = BatchEncoding(
            {
                'input_ids': torch.tensor(input_ids).to(device),
                'token_type_ids': torch.tensor(token_type_ids).to(device),
                'attention_mask': torch.tensor(attention_mask).to(device),
            }
        )
        return dict(
            id=row['id'],
            encodings=encodings,
            cls_token_ids=torch.tensor(cls_token_ids).to(device),
            ext_label=torch.tensor(ext_label).to(device)
        )
