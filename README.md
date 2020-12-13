# fine-tune-bert-for-text-classification
Code to reproduce the paper "How to Fine-Tune BERT for Text Classification"

### baseline

#### imdb dataset

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_baseline.py
```

- reported result (best test): 94.37% (error rate 5.63%)
- reproduced result (average over 3 seeds)
  - best test: 94.13% (0.05%)
  - last test: 94.12% (0.05%)
