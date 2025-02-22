import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def train(cfg: DictConfig):
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

    if cfg.model_checkpoint:
        model.load_state_dict(convert_checkpoints(cfg.model_checkpoint), strict=False)
        print("model checkpoint '{}' is loaded".format(cfg.model_checkpoint))

    # 3. load train and validation datasets
    if cfg.mode.dataset == 'cnn_dm':
        train_df, val_df = cnndm_train_df(cfg.dataset.path, **cfg.dataset.df)
    elif cfg.mode.dataset == 'pubmed':
        train_df, val_df = pubmed_train_df(cfg.dataset.path, **cfg.dataset.df)
    elif cfg.mode.dataset == 'wikihow':
        train_df, val_df = wikihow_train_df(cfg.dataset.path, **cfg.dataset.df)
    elif cfg.mode.dataset == 'xsum':
        train_df, val_df = xsum_train_df(cfg.dataset.path, **cfg.dataset.df)
    else:
        raise ValueError(
            "Please choose the dataset in cnn_dm, pubmed, wikihow, xsum, multinews")

    if 'bart' in cfg.mode.model:
        train_dataset = Bart_Dataset(
            train_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            cls_token=cls_token,
            sep_token=sep_token,
            doc_token=doc_token
        )
        val_dataset = Bart_Dataset(
            val_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            cls_token=cls_token,
            sep_token=sep_token,
            doc_token=doc_token
        )
    elif 'bert' in cfg.mode.model:
        train_dataset = Bert_Dataset(
            train_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            doc_token=doc_token
        )
        val_dataset = Bert_Dataset(
            val_df,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            doc_token=doc_token
        )
    else:
        raise ValueError("base model & tokenizer must be BertSum or BartSum.")

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False)

    # 4. config training
    if cfg.mode.model in ['bertsum', 'bartsum']:
        engine = BertSum_Engine(model, train_df, val_df, **cfg.engine)
    else:
        engine = OrderSum_Engine(model, train_df, val_df, **cfg.engine)

    logger = Another_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()

    # 5. run training
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    logger.watch(engine)

    if cfg.train_checkpoint:
        trainer.fit(engine, train_loader, val_loader, ckpt_path=cfg.train_checkpoint)
    else:
        trainer.fit(engine, train_loader, val_loader)

    wandb.finish()


if __name__ == "__main__":
    train()

