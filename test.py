import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def test(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # 1. prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    if 'bart' in cfg.mode.model:
        cls_token = "<cls>"
        sep_token = "<sep>"
        doc_token = "<doc>"
        new_tokens = [cls_token, sep_token, doc_token]

    elif 'bert' in cfg.mode.model:
        doc_token = "[DOC]"
        new_tokens = [doc_token]
    else:
        raise ValueError("base model & tokenizer must be BertSum or BartSum.")

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # 2. prepare model and loss function
    if cfg.mode.model == 'bartsum':
        model = BartSum_Ext(tokenizer, **cfg.model)
    elif cfg.mode.model == 'bertsum':
        model = BertSum_Ext(tokenizer, **cfg.model)
    elif cfg.mode.model == 'colo_bart':
        model = CoLo_BART(tokenizer, **cfg.model)
    elif cfg.mode.model == 'colo_bert':
        model = CoLo_BERT(tokenizer, **cfg.model)
    elif cfg.mode.model == 'ordersum_bart':
        model = OrderSum_BART(tokenizer, **cfg.model)
    elif cfg.mode.model == 'ordersum_bert':
        model = OrderSum_BERT(tokenizer, **cfg.model)
    else:
        raise ValueError(
            "Please choose the mode of model in bartsum, bertsum, colo_bart, colo_bert, ordersum_bart, ordersum_bert.")

    if cfg.test_checkpoint:
        model.load_state_dict(convert_checkpoints(cfg.test_checkpoint))
        print("model checkpoint '{}' is loaded".format(cfg.test_checkpoint))

    # 3. load train and validation datasets
    if cfg.mode.dataset == 'cnn_dm':
        test_df = cnndm_test_df(cfg.dataset.path)
    elif cfg.mode.dataset == 'pubmed':
        test_df = pubmed_test_df(cfg.dataset.path)
    elif cfg.mode.dataset == 'wikihow':
        test_df = wikihow_test_df(cfg.dataset.path)
    elif cfg.mode.dataset == 'xsum':
        test_df = xsum_test_df(cfg.dataset.path)
    else:
        raise ValueError(
            "Please choose the dataset in cnn_dm, pubmed, wikihow, xsum, multinews")

    if 'bart' in cfg.mode.model:
        test_dataset = Bart_Dataset(
            test_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            cls_token=cls_token,
            sep_token=sep_token,
            doc_token=doc_token
        )
    elif 'bert' in cfg.mode.model:
        test_dataset = Bert_Dataset(
            test_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            doc_token=doc_token
        )
    else:
        raise ValueError("base model & tokenizer must be BertSum or BartSum.")

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. run test
    if cfg.mode.model in ['bertsum', 'bartsum']:
        engine = BertSum_Engine(model, test_df=test_df, **cfg.engine)
    else:
        engine = OrderSum_Engine(model, test_df=test_df, **cfg.engine)

    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False)

    if 'test_checkpoint' in cfg:
        trainer.test(engine, test_loader, ckpt_path=cfg.test_checkpoint)
    else:
        raise RuntimeError('no checkpoint is given')


if __name__ == "__main__":
    test()
