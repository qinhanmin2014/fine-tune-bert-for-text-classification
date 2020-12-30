import argparse
import pickle
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForPreTraining
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-max_seq_length', default=128, type=int)
parser.add_argument('-max_predictions_per_seq', default=20, type=int)
parser.add_argument('-num_epochs', default=3, type=int)
parser.add_argument('-num_warmup_steps', default=10000, type=int)
parser.add_argument('-learning_rate', default=5e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-seed', default=0, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

with open("cache/imdb_further_pretraining", "rb") as reader:
    (train_input_ids, train_attention_masks, train_segment_ids,
     train_masked_lm_labels, train_next_sentence_labels) = pickle.load(reader) 

train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_masks = torch.tensor(train_attention_masks, dtype=torch.float)
train_segment_ids = torch.tensor(train_segment_ids, dtype=torch.long)
train_masked_lm_labels = torch.tensor(train_masked_lm_labels, dtype=torch.long)
train_next_sentence_labels = torch.tensor(train_next_sentence_labels, dtype=torch.long)
train_data = TensorDataset(train_input_ids, train_attention_masks, train_segment_ids,
                           train_masked_lm_labels, train_next_sentence_labels)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

model = BertForPreTraining.from_pretrained("bert-base-uncased")
model.to(device);

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.num_warmup_steps,
                num_training_steps=len(train_loader) * args.num_epochs)
total_step = len(train_loader)
for epoch in range(args.num_epochs):
    model.train()
    model.zero_grad()
    for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids,
            cur_masked_lm_labels, cur_next_sentence_labels) in enumerate(train_loader):
        cur_input_ids = cur_input_ids.to(device)
        cur_attention_mask = cur_attention_mask.to(device)
        cur_token_type_ids = cur_token_type_ids.to(device)
        cur_masked_lm_labels = cur_masked_lm_labels.to(device)
        cur_next_sentence_labels = cur_next_sentence_labels.to(device)
        loss = model(cur_input_ids, cur_attention_mask, cur_token_type_ids,
                     labels=cur_masked_lm_labels, next_sentence_label=cur_next_sentence_labels)[0]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))
    model.save_pretrained("imdb_further_pretraining_epoch_" + str(epoch))
