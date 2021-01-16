import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers.optimization_tf import WarmUp, AdamWeightDecay

parser = argparse.ArgumentParser()
parser.add_argument('-max_seq_length', default=512, type=int)
parser.add_argument('-batch_size', default=24, type=int)
parser.add_argument('-num_epochs', default=4, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-bert_path', default='bert-base-uncased')
parser.add_argument('-trunc_mode', default=128, type=str)
args = parser.parse_args()

def load_data(path):
    input_ids, attention_mask, token_type_ids = [], [], []
    sentiments = []
    input_file = open(path, encoding="utf8")
    lines = input_file.readlines()
    input_file.close()
    for line in tqdm(lines):
        label, text = line.split("\t")
        text = tokenizer.tokenize(text)
        if args.trunc_mode == "head":
            if len(text) > args.max_seq_length - 2:
                text = text[:args.max_seq_length - 2]
        elif args.trunc_mode == "tail":
            if len(text) > args.max_seq_length - 2:
                text = text[-(args.max_seq_length - 2):]
        else:
            args.trunc_mode = int(args.trunc_mode)
            assert args.trunc_mode < args.max_seq_length
            if len(text) > args.max_seq_length - 2:
                text = text[:args.trunc_mode] + text[-(args.max_seq_length - 2 - args.trunc_mode):]
        text = ["[CLS]"] + text + ["[SEP]"]
        attention_mask.append([1] * len(text) + [0] * (args.max_seq_length - len(text)))
        token_type_ids.append([0] * args.max_seq_length)
        input_ids.append(tokenizer.convert_tokens_to_ids(text) + [0] * (args.max_seq_length - len(text)))
        sentiments.append(int(label))
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(sentiments)

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
train_path = os.path.join("data/imdb", 'train.csv')
test_path = os.path.join("data/imdb", 'test.csv')
train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(train_path)
test_input_ids, test_attention_mask, test_token_type_ids, y_test = load_data(test_path)

num_train_steps = train_input_ids.shape[0] * args.num_epochs // args.batch_size
num_warmup_steps = int(num_train_steps * args.warm_up_proportion)
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=0
    )
    lr_schedule = WarmUp(
        initial_learning_rate=args.learning_rate,
        decay_schedule_fn=lr_schedule,
        warmup_steps=num_warmup_steps
    )
    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        #clipnorm=1
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit([train_input_ids, train_attention_mask, train_token_type_ids], y_train,
          validation_data=([test_input_ids, test_attention_mask, test_token_type_ids], y_test),
          batch_size=args.batch_size, epochs=args.num_epochs)
