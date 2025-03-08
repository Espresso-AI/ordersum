# OrderSum: Semantic Sentence Ordering for Extractive Summarization

Official implementation for [OrderSum: Semantic Sentence Ordering for Extractive Summarization](https://www.arxiv.org/abs/2502.16180)

> *There are two main approaches to recent extractive summarization: the sentence-level framework, which selects sentences to include in a summary individually, and the summary-level framework, which generates multiple candidate summaries and ranks them. Previous work in both frameworks has primarily focused on improving which sentences in a document should be included in the summary. However, the sentence order of extractive summaries, which is critical for the quality of a summary, remains underexplored. In this paper, we introduce OrderSum, a novel extractive summarization model that semantically orders sentences within an extractive summary. OrderSum proposes a new representation method to incorporate the sentence order into the embedding of the extractive summary, and an objective function to train the model to identify which extractive summary has a better sentence order in the semantic space. Extensive experimental results demonstrate that OrderSum obtains state-of-the-art performance in both sentence inclusion and sentence order for extractive summarization. In particular, OrderSum achieves a ROUGE-L score of 30.52 on CNN/DailyMail, outperforming the previous state-of-the-art model by a large margin of 2.54.*

## Dependencies
The code is developed based on PyTorch, Transformers, and PyTorch Lightning.

We use WandB for tracking model training and Hydra for managing experiment configurations.

To evaluate the sentence order of summaries, we use [rouge-score](https://pypi.org/project/rouge-score/) from Google-Research. For more details, please refer to Appendix 1 of the paper.

- python : 3.9
- torch : 1.13.1
- transformers : 4.34.0
- pytorch-lightning : 1.9.0
- rouge-score : 0.1.2

## Dataset
We conduct experiments on four datasets, CNN/DailyMail, XSum, WikiHow, and PubMed.

You can download the preprocessed datasets from [here](https://drive.google.com/drive/folders/1E5K5pte7n3tduz_TMOEaYaPzSPGaWqW8).

## Training
Run the following command to train the model on CNN/DailyMail:

```
python train.py --config-name cnndm_ordersum_1024
```

The experiment configuration is specified by the yaml file in `./config`. When passing the configuration, the `.yaml` extension should be removed.

The dataset path, validation ratio, etc. of each dataset is configured in `./config/dataset`. The configuration file for PyTorch Lightning Trainer is located in `./config/trainer`.

We trained our model on a single Nvidia Tesla T4.

## Evaluation
You can download [OrderSum checkpoints](https://drive.google.com/drive/folders/1A_wtdsP81s-Fpck59oUG22ws0ek39pvh) trained on the four datasets.

On CNN/DailyMail, OrderSum significantly outperforms the existing models on ROUGE-L, achieving state-of-the-art performance.

|                    |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  |
|--------------------|:---------:|:---------:|:---------:|
| BERTSUM            |   42.90   |   20.06   |   27.62   |
| BARTSUM            |   43.81   |   20.85   |   28.09   |
| BARTSUM 1024       |   44.09   |   21.09   |   28.34   |
| MatchSum + BERT    |   43.98   |   20.50   |   28.62   |
| MatchSum + RoBERTa |   44.17   |   20.80   |   28.86   |
| CoLo               |   44.18   |   21.03   |   27.98   |
| CoLo 1024          |   44.41   |   21.19   |   27.98   |
| OrderSum           |   44.23   |   21.17   |   30.21   |
| **OrderSum 1024**  | **44.44** | **21.31** | **30.52** |

Since the official repository of [CoLo](https://github.com/ChenxinAn-fdu/CoLo) did not release checkpoints or result files, we retrained CoLo to evaluate using rouge-score.

You can download our version of [CoLo checkpoints](https://drive.google.com/drive/folders/1M2L_5SF_CrJIrh3RlF46wYx7uCPkowYh), which obtains higher scores than those reported in the paper when evaluated using pyrouge.

Run the following command to test the model on CNN/DailyMail:

```
python test.py --config-name cnndm_ordersum_1024
```

Ensure that the checkpoint file path to be tested is specified in `test_checkpoint` in the yaml file.

## License
BSD 3-Clause License Copyright (c) 2022