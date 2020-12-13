import argparse
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default='imdb', choices=['imdb'])
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-max_seq_length', default=512, type=int)
parser.add_argument('-batch_size', default=24, type=int)
parser.add_argument('-num_epochs', default=4, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-gradient_accumulation_step', default=1, type=int)
parser.add_argument('-bert_path', default='bert-base-uncased')
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


tokenizer = BertTokenizer.from_pretrained(args.bert_path)
model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
model = torch.nn.DataParallel(model)
model.to(device);


def load_data(path):
    indices, sentiments = [], []
    input_file = open(path, encoding="utf8")
    line = input_file.readline()
    while line:
        label, text = line.split("\t")
        indices.append(tokenizer.encode(text, max_length=args.max_seq_length, padding="max_length", truncation=True))
        sentiments.append(int(label))
        line = input_file.readline()
    input_file.close()
    return np.array(indices), np.array(sentiments)


train_path = os.path.join("data/imdb", 'train.csv')
test_path = os.path.join("data/imdb", 'test.csv')
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)


X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion,
                num_training_steps=len(train_loader) * args.num_epochs)
total_step = len(train_loader)
for epoch in range(args.num_epochs):
    model.train()
    model.zero_grad()
    for i, (cur_X_train, cur_y_train) in enumerate(train_loader):
        cur_X_train = cur_X_train.to(device)
        cur_y_train = cur_y_train.to(device)
        outputs = model(cur_X_train)
        loss = nn.CrossEntropyLoss()(outputs[0], cur_y_train)
        loss /= args.gradient_accumulation_step
        loss.backward()
        if (i + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        if (i + 1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for cur_X_test, cur_y_test in tqdm(test_loader):
            cur_X_test = cur_X_test.to(device)
            cur_y_test = cur_y_test.to(device)
            outputs = model(cur_X_test)
            _, predicted = torch.max(outputs[0], 1)
            total += cur_y_test.size(0)
            correct += (predicted == cur_y_test).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))
