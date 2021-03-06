# fine-tune-bert-for-text-classification
Code to reproduce the paper "How to Fine-Tune BERT for Text Classification"

- original paper: https://arxiv.org/abs/1905.05583
- original repo: https://github.com/xuyige/BERT4doc-Classification

The imdb dataset used in the paper is preprocessed by the author (e.g., HTML tags are removed), and can be obtained from https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM.

### baseline (imdb dataset)

```
python bert_baseline.py
```

- reported result (best test): 94.37% (error rate 5.63%)
- reproduced result (average over 3 seeds)
  - best test: 94.20% (0.08%)
  - last test: 94.19% (0.08%)

### 5.3.1 Dealing with long texts (imdb dataset)

```
python bert_5_3_1.py -max_seq_length XXX -trunc_mode head
python bert_5_3_1.py -max_seq_length XXX -trunc_mode tail
python bert_5_3_1.py -max_seq_length XXX -trunc_mode 128
```

- head-only, max_seq_length=512
  - reported result (best test): 94.37% (error rate 5.63%)
  - reproduced result (average over 3 seeds)
    - best test: 94.20% (0.08%)
    - last test: 94.19% (0.08%)
- tail-only, max_seq_length=512
  - reported result (best test): 94.56% (error rate 5.44%)
  - reproduced result (average over 3 seeds)
    - best test: 94.48% (0.04%)
    - last test: 94.47% (0.05%)
- head+tail, max_seq_length=512
  - reported result (best test): 94.58% (error rate 5.42%)
  - reproduced result (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
- head-only, max_seq_length=256
  - reproduced result (average over 3 seeds)
    - best test: 92.57% (0.06%)
    - last test: 92.52% (0.06%)
- tail-only, max_seq_length=256
  - reproduced result (average over 3 seeds)
    - best test: 93.97% (0.03%)
    - last test: 93.97% (0.03%)
- head+tail, max_seq_length=256
  - reproduced result (average over 3 seeds)
    - best test: 94.01% (0.04%)
    - last test: 93.96% (0.09%)
- **Conclusion: head + tail is usually the best way.**

### 5.3.2 Features from Different layers (imdb dataset)

```
python bert_5_3_2.py -pool_mode concat
python bert_5_3_2.py -pool_mode mean
python bert_5_3_2.py -pool_mode max
```
- baseline (head+tail, max_seq_length=512)
  - reported result (best test): 94.58% (error rate 5.42%)
  - reproduced result (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
- Last 4 Layers + concat
  - reported result (best test): 94.57% (error rate 5.43%)
  - reproduced result (average over 3 seeds)
    - best test: 94.50% (0.03%)
    - last test: 94.50% (0.03%)
- Last 4 Layers + mean
  - reported result (best test): 94.56% (error rate 5.44%)
  - reproduced result (average over 3 seeds)
    - best test: 94.42% (0.07%)
    - last test: 94.39% (0.03%)
- Last 4 Layers + max
  - reported result (best test): 94.58% (error rate 5.42%)
  - reproduced result (average over 3 seeds)
    - best test: 94.53% (0.08%)
    - last test: 94.50% (0.07%)
- **Conclusion: Pooling the first output state of the last several layers won't lead to better results on imdb dataset.**

### 5.3.3 Catastrophic Forgetting (imdb dataset)

```
python bert_5_3_1.py -learning_rate XXX
```

- head+tail, max_seq_length=512, learning_rate 2.0e-5 (average over 3 seeds)
  - best test: 94.57% (0.07%)
  - last test: 94.49% (0.04%)
- head+tail, max_seq_length=512, learning_rate 5.0e-5 (average over 3 seeds)
  - best test: 94.30% (0.12%)
  - last test: 94.27% (0.13%)
- head+tail, max_seq_length=512, learning_rate 1.0e-4 (average over 3 seeds)
  - best test: 91.95% (0.32%)
  - last test: 91.95% (0.32%)
  - loss won't decrease to a small value
- head+tail, max_seq_length=512, learning_rate 4.0e-4 (average over 3 seeds)
  - best test: 50.00% (0.00%)
  - last test: 50.00% (0.00%)
  - loss won't decrease
- **Conclusion: A small learning rate is usually better.**

### 5.3.4 Layer-wise Decreasing Layer Rate (imdb dataset)

```
python bert_5_3_4.py -learning_rate_decay XXX -learning_rate XXX
```

- baseline (learning rate 2.0e-5, head+tail, max_seq_length=512)
  - reported result (best test): 94.58% (error rate 5.42%)
  - reproduced result (average over 3 seeds)
    - best test: 94.51% (0.08%)
    - last test: 94.45% (0.06%)
- learning rate 2.0e-5, decay factor 0.95
  - reported result (best test): 94.60% (error rate 5.40%)
  - reproduced result (average over 3 seeds)
    - best test: 94.60% (0.07%)
    - last test: 94.55% (0.01%)
- learning rate 2.0e-5, decay factor 0.90
  - reported result (best test): 94.48% (error rate 5.52%)
  - reproduced result (average over 3 seeds)
    - best test: 94.54% (0.06%)
    - last test: 94.53% (0.06%)
- learning rate 2.0e-5, decay factor 0.85
  - reported result (best test): 94.35% (error rate 5.65%)
  - reproduced result (average over 3 seeds)
    - best test: 94.46% (0.12%)
    - last test: 94.46% (0.12%)
- learning rate 2.5e-5, decay factor 1.00
  - reported result (best test): 94.48% (error rate 5.52%)
  - reproduced result (average over 3 seeds)
    - best test: 94.54% (0.05%)
    - last test: 94.53% (0.05%)
- learning rate 2.5e-5, decay factor 0.95
  - reported result (best test): 94.54% (error rate 5.46%)
  - reproduced result (average over 3 seeds)
    - best test: 94.64% (0.06%)
    - last test: 94.59% (0.10%)
- learning rate 2.5e-5, decay factor 0.90
  - reported result (best test): 94.56% (error rate 5.44%)
  - reproduced result (average over 3 seeds)
    - best test: 94.50% (0.05%)
    - last test: 94.50% (0.05%)
- learning rate 2.5e-5, decay factor 0.85
  - reported result (best test): 94.42% (error rate 5.58%)
  - reproduced result (average over 3 seeds)
    - best test: 94.46% (0.02%)
    - last test: 94.46% (0.02%)
- **Conclusion: A small decay factor (e.g., 0.95) usually leads to better results.**

### 5.4 Investigating the Further Pretraining (imdb dataset)

```
python bert_5_4_create_pretraining_data.py
python bert_5_4_run_pretraining.py
python bert_5_3_1.py -seed 0 -bert_path imdb_further_pretraining_epoch_X/
```

- baseline (learning rate 2.0e-5, head+tail, max_seq_length=512)
  - reported result (best test): 94.58% (error rate 5.42%)
  - reproduced result (average over 3 seeds)
    - best test: 94.51% (0.08%)
    - last test: 94.45% (0.06%)
- fine tuning using training set
  - reported result (best test): 95.63% (error rate 4.37%)
  - reproduced result (epoch 3 model, average over 3 seeds)
    - best test: 95.19% (0.03%)
    - last test: 95.10% (0.08%)
  - reproduced result (epoch 2 model, average over 3 seeds)
    - best test: 95.19% (0.07%)
    - last test: 95.07% (0.07%)
  - reproduced result (epoch 1 model, average over 3 seeds)
    - best test: 95.00% (0.08%)
    - last test: 94.98% (0.10%)
- **Conclusion: Further Pretraining usually leads to better results.**

### 5.5 Multi-task Fine-Tuning (imdb dataset)

```
python bert_5_5.py
```

- yelp p dataset: https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
- baseline
  - train using imdb training set (average over 3 seeds)
    - best test: 94.20% (0.08%)
    - last test: 94.19% (0.08%)
  - train using imdb training set + yelp p test set (average over 3 runs)
    - best test: 94.46% (0.03%)
    - last test: 94.43% (0.07%)
- head+tail, max_seq_length=512
  - train using imdb training set (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
  - train using imdb training set + yelp p test set (average over 3 runs)
    - best test: 94.71% (0.07%)
    - last test: 94.67% (0.07%)
- **Conclusion: Multi-task fine tuning can leads to better results.**

### extra experiment: influence of max lengths (imdb dataset)

```
python bert_baseline.py -max_seq_length XXX
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
- **Conclusion: When some texts exceed max_seq_length, larger max_seq_length is usually better.**

### extra experiment: pool the output states instead of taking the first output state (imdb dataset)

```
python bert_pool_last_layer.py -pool_mode mean -num_pool_layers XXX
python bert_pool_last_layer.py -pool_mode max -num_pool_layers XXX
python bert_pool_last_layer.py -pool_mode mean -num_pool_layers XXX -bert_path roberta-base
python bert_pool_last_layer.py -pool_mode max -num_pool_layers XXX -bert_path roberta-base
```

- bert
  - baseline (head+tail, max_seq_length=512, average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
  - average pooling, last layer
    - best test: 94.47% (0.11%)
    - last test: 94.44% (0.14%)
  - average pooling, concat last 3 layers
    - best test: 94.48% (0.06%)
    - last test: 94.47% (0.07%)
  - max pooling, last layer
    - best test: 94.43% (0.12%)
    - last test: 94.41% (0.10%)
  - max pooling, concat last 3 layers
    - best test: 94.45% (0.08%)
    - last test: 94.41% (0.13%)
- roberta
  - baseline (head+tail, max_seq_length=512, average over 3 seeds)
    - best test: 95.66% (0.05%)
    - last test: 95.66% (0.05%)
  - average pooling, last layer
    - best test: 95.81% (0.04%)
    - last test: 95.81% (0.04%)
  - average pooling, concat last 3 layers
    - best test: 95.57% (0.13%)
    - last test: 95.56% (0.11%)
  - max pooling, last layer
    - best test: 95.68% (0.03%)
    - last test: 95.65% (0.02%)
  - max pooling, concat last 3 layers
    - best test: 95.71% (0.07%)
    - last test: 95.67% (0.05%)
- **Conclusion: Pooling the output states can lead to better results if we use roberta.**

### extra experiment: influence of gradient accumulation (imdb dataset)

```
python bert_5_3_1.py -batch_size 12 -gradient_accumulation_step 2
```

- baseline (head+tail, max_seq_length=512, average over 3 seeds)
  - best test: 94.57% (0.07%)
  - last test: 94.49% (0.04%)
- gradient accumulation (average over 3 seeds)
  - best test: 94.60% (0.08%)
  - last test: 94.60% (0.08%)
- **Conclusion: We can get similar results with less GPU memory through gradient accumulation.**

### extra experiment: influence of data preprocessing (imdb dataset)

```
python bert_data_preprocessing.py
```

- baseline (processed dataset used in the paper, head+tail, max_seq_length=512, average over 3 seeds)
  - best test: 94.57% (0.07%)
  - last test: 94.49% (0.04%)
- original dataset (head+tail, max_seq_length=512, average over 3 seeds)
  - best test: 94.51% (0.02%)
  - last test: 94.47% (0.05%)
- baseline (processed dataset used in the paper, head+tail, max_seq_length=256, average over 3 seeds)
  - best test: 94.01% (0.04%)
  - last test: 93.96% (0.09%)
- original dataset (head+tail, max_seq_length=256, average over 3 seeds)
  - best test: 94.12% (0.07%)
  - last test: 94.03% (0.11%)
- **Conclusion: Data preprocessing has little influence on the performance of bert.**

### extra experiment: average the outputs of different models (imdb dataset)

- baseline, single model, 5 runs: mean 94.53%, max 94.676%
  - soft vote
    - 1 models: 94.292%
    - 2 models: 94.496%
    - 3 models: 94.504%
    - 4 models: 94.652%
    - 5 models: 94.656%
  - hard vote
    - 1 models: 94.292%
    - 2 models: 94.392%
    - 3 models: 94.504%
    - 4 models: 94.568%
    - 5 models: 94.676%
- head+tail, max_seq_length=512, single model, 5 runs: mean 94.27%, max 94.392%
  - soft vote
    - 1 models: 94.676%
    - 2 models: 94.752%
    - 3 models: 94.888%
    - 4 models: 94.916%
    - 5 models: 94.94%
  - hard vote
    - 1 models: 94.676%
    - 2 models: 94.64%
    - 3 models: 94.896%
    - 4 models: 94.852%
    - 5 models: 94.944%
- **Conclusion: Averaging the outputs of different models can improve the results.**

### extra experiment: other pretrained models (imdb dataset)

```
python bert_other_models.py -bert_path roberta-base
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_5_3_1.py -batch_size 8 -gradient_accumulation_step 3 -bert_path bert-large-uncased
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_other_models.py -batch_size 8 -gradient_accumulation_step 3 -bert_path roberta-large
```

- baseline
  - bert result (average over 3 seeds)
    - best test: 94.20% (0.08%)
    - last test: 94.19% (0.08%)
  - roberta result (average over 3 runs)
    - best test: 95.55% (0.04%)
    - last test: 95.55% (0.04%)
- head+tail, max_seq_length=512
  - bert result (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
  - bert-large-uncased result (average over 3 seeds)
    - best test: 95.28% (0.03%)
    - last test: 95.32% (0.02%)
  - roberta result (average over 3 runs)
    - best test: 95.66% (0.05%)
    - last test: 95.66% (0.05%)
  - roberta-large result (average over 3 seeds)
    - best test: 96.39% (0.06%)
    - last test: 96.32% (0.10%)
- **Conclusion: We can often replace bert with roberta to get better results.**

### extra experiment: tensorflow instead of pytorch (imdb dataset)

```
python tf_bert_5_3_1.py
```

- baseline
  - pytorch result (average over 3 seeds)
    - best test: 94.20% (0.08%)
    - last test: 94.19% (0.08%)
  - tensorflow result (average over 3 runs)
    - best test: 94.16% (0.03%)
    - last test: 94.16% (0.03%)
- head+tail, max_seq_length=512
  - pytorch result (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
  - tensorflow result (average over 3 runs)
    - best test: 94.53% (0.02%)
    - last test: 94.52% (0.00%)
- **Conclusion: We can get similar results using tensorflow.**

### extra experiment: tpu instead of gpu (imdb dataset)

```
tf_bert_5_3_1_kaggle_tpu.ipynb (use tensorflow & tpu provided by kaggle)
pytorch_bert_5_3_1_kaggle_tpu.ipynb (use pytorch & tpu provided by kaggle)
tf_bert_5_3_1_colab_tpu.ipynb (use tensorflow & tpu provided by colab)
pytorch_bert_5_3_1_colab_tpu.ipynb (use pytorch & tpu provided by colab)
```

- kaggle tpu is better than colab tpu (e.g., can support larger batch_size) so the results below are from kaggle tpu
- baseline
  - pytorch result (average over 3 seeds)
    - best test: 94.20% (0.08%)
    - last test: 94.19% (0.08%)
  - tensorflow tpu result (average over 3 runs)
    - best test: 94.24% (0.08%)
    - last test: 94.17% (0.10%)
  - pytorch tpu result (average over 3 seeds)
    - best test: 94.10% (0.05%)
    - last test: 94.10% (0.05%)
- head+tail, max_seq_length=512
  - pytorch result (average over 3 seeds)
    - best test: 94.57% (0.07%)
    - last test: 94.49% (0.04%)
  - tensorflow tpu result (average over 3 runs)
    - best test: 94.51% (0.06%)
    - last test: 94.44% (0.13%)
  - pytorch tpu result (average over 3 seeds)
    - best test: 94.54% (0.08%)
    - last test: 94.53% (0.09%)
- **Conclusion: We can get similar results using tpu.**
