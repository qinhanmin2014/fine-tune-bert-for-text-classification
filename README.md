# fine-tune-bert-for-text-classification
Code to reproduce the paper "How to Fine-Tune BERT for Text Classification"

- original paper: https://arxiv.org/abs/1905.05583
- original repo: https://github.com/xuyige/BERT4doc-Classification

### baseline (imdb dataset)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_baseline.py
```

- reported result (best test): 94.37% (error rate 5.63%)
- reproduced result (average over 3 seeds)
  - best test: 94.20% (0.08%)
  - last test: 94.19% (0.08%)

### 5.3.1 Dealing with long texts (imdb dataset)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_5_3_1.py -max_seq_length XXX -trunc_mode head
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_5_3_1.py -max_seq_length XXX -trunc_mode tail
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_5_3_1.py -max_seq_length XXX -trunc_mode 128
```

- head-only, max_seq_length=512 (average over 3 seeds)
    - reported result (best test): 94.37% (error rate 5.63%)
    - reproduced result (average over 3 seeds)
      - best test: 94.13% (0.05%)
      - last test: 94.12% (0.05%)
- tail-only, max_seq_length=512 (average over 3 seeds)
    - reported result (best test): 94.56% (error rate 5.44%)
    - reproduced result (average over 3 seeds)
      - best test: 94.12% (0.11%)
      - last test: 94.12% (0.11%)
- head+tail, max_seq_length=512 (average over 3 seeds)
    - reported result (best test): 94.58% (error rate 5.42%)
    - reproduced result (average over 3 seeds)
      - best test: 94.12% (0.07%)
      - last test: 94.12% (0.07%)
- head-only, max_seq_length=256 (average over 3 seeds)
    - reproduced result (average over 3 seeds)
      - best test: 92.43% (0.08%)
      - last test: 92.39% (0.07%)
- tail-only, max_seq_length=256 (average over 3 seeds)
    - reproduced result (average over 3 seeds)
      - best test: 93.88% (0.01%)
      - last test: 93.88% (0.01%)
- head+tail, max_seq_length=256 (average over 3 seeds)
    - reproduced result (average over 3 seeds)
      - best test: 93.99% (0.05%)
      - last test: 93.98% (0.05%)
- Conclusion: head + tail is the best way, but when max_seq_length is large enough, the difference is small.

### extra experiment: influence of max lengths (imdb dataset)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_baseline.py -max_seq_length XXX
```

- basic info
  - average length: training set 295.57, test set 288.75
  - median length: training set 220, test set 217
  - max length: training set 3045, test set 2873
- max_seq_length=128 (average over 3 seeds)
  - exceeding ratio: training set 87.91%, test set 87.42%
  - best test: 89.40% (0.06%)
  - last test: 89.20% (0.10%)
- max_seq_length=256 (average over 3 seeds)
  - exceeding ratio: training set 41.14%, test set 40.11%
  - best test: 92.57% (0.06%)
  - last test: 92.52% (0.06%)
- max_seq_length=512 (original baseline, average over 3 seeds)
  - exceeding ratio: training set 13.28%, test set 12.34%
  - best test: 94.20% (0.08%)
  - last test: 94.19% (0.08%)
- Conclusion: When some texts exceed max_seq_length, larger max_seq_length is better.
