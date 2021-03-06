import argparse
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-max_seq_length', default=512, type=int)
parser.add_argument('-batch_size', default=24, type=int)
parser.add_argument('-num_epochs', default=4, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-gradient_accumulation_step', default=1, type=int)
parser.add_argument('-bert_path', default='bert-base-uncased', type=str)
parser.add_argument('-trunc_mode', default=128, type=str)
parser.add_argument('-num_pool_layers', default=1, type=int)
parser.add_argument('-pool_mode', default="mean", type=str)
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * args.num_pool_layers, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        selected_outputs = []
        for i in range(-args.num_pool_layers, 0):
            selected_outputs.append(outputs[2][i])
        selected_outputs = torch.cat(selected_outputs, dim=2)
        if args.pool_mode == "mean":
            pooled_output = torch.mean(selected_outputs, dim=1)
        elif args.pool_mode == "max":
            pooled_output, _ = torch.max(selected_outputs, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * args.num_pool_layers, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if args.pool_mode == "mean":
            x = torch.mean(features, dim=1)
        elif args.pool_mode == "max":
            x, _ = torch.max(features, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = []
        for i in range(-args.num_pool_layers, 0):
            sequence_output.append(outputs.hidden_states[i])
        sequence_output = torch.cat(sequence_output, dim=2)
        logits = self.classifier(sequence_output)
        return logits


if args.bert_path == "bert-base-uncased":
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
elif args.bert_path == "roberta-base":
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_path)
    model = RobertaForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
model = torch.nn.DataParallel(model)
model.to(device);


def load_data(path):
    input_ids, attention_mask, token_type_ids = [], [], []
    sentiments = []
    input_file = open(path, encoding="utf8")
    line = input_file.readline()
    while line:
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
        if args.bert_path == "bert-base-uncased":
            text = ["[CLS]"] + text + ["[SEP]"]
        elif args.bert_path == "roberta-base":
            text = ["<s>"] + text + ["</s>"]
        attention_mask.append([1] * len(text) + [0] * (args.max_seq_length - len(text)))
        token_type_ids.append([0] * args.max_seq_length)
        if args.bert_path == "bert-base-uncased":
            input_ids.append(tokenizer.convert_tokens_to_ids(text) + [0] * (args.max_seq_length - len(text)))
        elif args.bert_path == "roberta-base":
            input_ids.append(tokenizer.convert_tokens_to_ids(text) + [1] * (args.max_seq_length - len(text)))
        sentiments.append(int(label))
        line = input_file.readline()
    input_file.close()
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(sentiments)


train_path = os.path.join("data/imdb", 'train.csv')
test_path = os.path.join("data/imdb", 'test.csv')
train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(train_path)
test_input_ids, test_attention_mask, test_token_type_ids, y_test = load_data(test_path)


train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.float)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.float)
test_token_type_ids = torch.tensor(test_token_type_ids, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
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
        if args.bert_path == "bert-base-uncased":
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
        elif args.bert_path == "roberta-base":
            outputs = model(cur_input_ids, cur_attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, cur_y)
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
            _, predicted = torch.max(outputs, 1)
            total += cur_y.size(0)
            correct += (predicted == cur_y).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))
