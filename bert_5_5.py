import argparse
import pickle
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-max_seq_length', default=512, type=int)
parser.add_argument('-batch_size', default=24, type=int)
parser.add_argument('-num_epochs', default=3, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-gradient_accumulation_step', default=1, type=int)
parser.add_argument('-bert_path', default='bert-base-uncased')
parser.add_argument('-trunc_mode', default=128, type=str)
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


tokenizer = BertTokenizer.from_pretrained(args.bert_path)
model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
model = torch.nn.DataParallel(model)
model.to(device);


def tokenize_text(text):
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
    cur_attention_mask = [1] * len(text) + [0] * (args.max_seq_length - len(text))
    cur_token_type_ids = [0] * args.max_seq_length
    cur_input_ids = tokenizer.convert_tokens_to_ids(text) + [0] * (args.max_seq_length - len(text))
    return cur_input_ids, cur_attention_mask, cur_token_type_ids


def load_data(path, dataset):
    input_ids, attention_mask, token_type_ids = [], [], []
    labels = []
    if dataset == "imdb":
        input_file = pd.read_csv(path, header=None, sep="\t").values
    else:
        input_file = pd.read_csv(path, header=None, sep=",").values
    for label, text in tqdm(input_file):
        cur_input_ids, cur_attention_mask, cur_token_type_ids = tokenize_text(text)
        input_ids.append(cur_input_ids)
        attention_mask.append(cur_attention_mask)
        token_type_ids.append(cur_token_type_ids)
        if dataset == "imdb":
            labels.append(int(label))
        else:
            labels.append(int(label) - 1)
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(labels)


try:
    cached_features_file = os.path.join("cache", "imdb_train_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "rb") as reader:
        imdb_train_input_ids, imdb_train_attention_mask, imdb_train_token_type_ids, imdb_y_train = pickle.load(reader)
    cached_features_file = os.path.join("cache", "imdb_test_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "rb") as reader:
        imdb_test_input_ids, imdb_test_attention_mask, imdb_test_token_type_ids, imdb_y_test = pickle.load(reader)
except Exception:   
    imdb_train_input_ids, imdb_train_attention_mask, imdb_train_token_type_ids, imdb_y_train = load_data(
        os.path.join("data/imdb", 'train.csv'), "imdb")
    imdb_test_input_ids, imdb_test_attention_mask, imdb_test_token_type_ids, imdb_y_test = load_data(
        os.path.join("data/imdb", 'test.csv'), "imdb")
    cached_features_file = os.path.join("cache", "imdb_train_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "wb") as writer:
        pickle.dump((imdb_train_input_ids, imdb_train_attention_mask, imdb_train_token_type_ids, imdb_y_train), writer)
    cached_features_file = os.path.join("cache", "imdb_test_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "wb") as writer:
        pickle.dump((imdb_test_input_ids, imdb_test_attention_mask, imdb_test_token_type_ids, imdb_y_test), writer)
try:
    # cached_features_file = os.path.join("cache", "yelp_train_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    # with open(cached_features_file, "rb") as reader:
    #     yelp_train_input_ids, yelp_train_attention_mask, yelp_train_token_type_ids, yelp_y_train = pickle.load(reader)
    cached_features_file = os.path.join("cache", "yelp_test_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "rb") as reader:
        yelp_test_input_ids, yelp_test_attention_mask, yelp_test_token_type_ids, yelp_y_test = pickle.load(reader)
except Exception:
    # yelp_train_input_ids, yelp_train_attention_mask, yelp_train_token_type_ids, yelp_y_train = load_data(
    #     os.path.join("data/yelp_review_polarity_csv", 'train.csv'), "yelp")
    yelp_test_input_ids, yelp_test_attention_mask, yelp_test_token_type_ids, yelp_y_test = load_data(
        os.path.join("data/yelp_review_polarity_csv", 'test.csv'), "yelp")
    # cached_features_file = os.path.join("cache", "yelp_train_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    # with open(cached_features_file, "wb") as writer:
    #     pickle.dump((yelp_train_input_ids, yelp_train_attention_mask, yelp_train_token_type_ids, yelp_y_train), writer)
    cached_features_file = os.path.join("cache", "yelp_test_" + str(args.max_seq_length) + "_" + str(args.trunc_mode))
    with open(cached_features_file, "wb") as writer:
        pickle.dump((yelp_test_input_ids, yelp_test_attention_mask, yelp_test_token_type_ids, yelp_y_test), writer)


# use test set of yelp dataset as extra data because
# (1) the topic of the two datasets are similar (sentiment classification)
# (2) the size of the two datasets are similar (imdb 25000, yelp 38000)
train_input_ids = torch.tensor(np.concatenate([imdb_train_input_ids, yelp_test_input_ids], axis=0), dtype=torch.long)
train_attention_mask = torch.tensor(np.concatenate([imdb_train_attention_mask, yelp_test_attention_mask], axis=0), dtype=torch.float)
train_token_type_ids = torch.tensor(np.concatenate([imdb_train_token_type_ids, yelp_test_token_type_ids], axis=0), dtype=torch.long)
y_train = torch.tensor(np.concatenate([imdb_y_train, yelp_y_test], axis=0), dtype=torch.long)
test_input_ids = torch.tensor(imdb_test_input_ids, dtype=torch.long)
test_attention_mask = torch.tensor(imdb_test_attention_mask, dtype=torch.float)
test_token_type_ids = torch.tensor(imdb_test_token_type_ids, dtype=torch.long)
y_test = torch.tensor(imdb_y_test, dtype=torch.long)
train_data = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, y_train)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, y_test)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion // args.gradient_accumulation_step,
                num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_step)
total_step = len(train_loader)
for epoch in range(args.num_epochs):
    model.train()
    model.zero_grad()
    for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(train_loader):
        cur_input_ids = cur_input_ids.to(device)
        cur_attention_mask = cur_attention_mask.to(device)
        cur_token_type_ids = cur_token_type_ids.to(device)
        cur_y = cur_y.to(device)
        outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
        loss = nn.CrossEntropyLoss()(outputs[0], cur_y)
        loss /= args.gradient_accumulation_step
        loss.backward()
        if (i + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(test_loader):
            cur_input_ids = cur_input_ids.to(device)
            cur_attention_mask = cur_attention_mask.to(device)
            cur_token_type_ids = cur_token_type_ids.to(device)
            cur_y = cur_y.to(device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            _, predicted = torch.max(outputs[0], 1)
            total += cur_y.size(0)
            correct += (predicted == cur_y).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))
