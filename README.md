# fine-tune-bert-for-text-classification
Code to reproduce the paper "How to Fine-Tune BERT for Text Classification"

- original paper: https://arxiv.org/abs/1905.05583
- original repo: https://github.com/xuyige/BERT4doc-Classification

### baseline

#### imdb dataset

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_baseline.py
```

- reported result (best test): 94.37% (error rate 5.63%)
- reproduced result (average over 3 seeds)
  - best test: 94.13% (0.05%)
  - last test: 94.12% (0.05%)
  
### extra experiment: influence of max lengths

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_baseline.py -max_seq_length XXX
```

#### imdb dataset
- basic info
  - average length: training set 295.57, test set 288.75
  - median length: training set 220, test set 217
  - max length: training set 3045, test set 2873
- max_seq_length=128 (average over 3 seeds)
  - exceeding ratio: training set 87.91%, test set 87.42%
  - best test: 89.30% (0.05%)
  - last test: 89.28% (0.04%)
- max_seq_length=256 (average over 3 seeds)
  - exceeding ratio: training set 41.14%, test set 40.11%
  - best test: 92.43% (0.08%)
  - last test: 92.39% (0.07%)
- max_seq_length=512 (original baseline, average over 3 seeds)
  - exceeding ratio: training set 13.28%, test set 12.34%
  - best test: 94.13% (0.05%)
  - last test: 94.12% (0.05%)
