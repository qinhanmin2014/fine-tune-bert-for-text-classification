# reference https://github.com/google-research/bert/blob/master/create_pretraining_data.py
import argparse
import pickle
from nltk import sent_tokenize
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-bert_path', default='bert-base-uncased')
parser.add_argument('-short_seq_prob', default=0.1, type=float)
parser.add_argument('-max_seq_length', default=128, type=int)
parser.add_argument('-max_predictions_per_seq', default=20, type=int)
parser.add_argument('-masked_lm_prob', default=0.15, type=float)
parser.add_argument('-dupe_factor', default=10, type=int)
args = parser.parse_args()

all_documents = []
input_file = open("data/imdb/train.csv", encoding="utf8")
line = input_file.readline()
while line:
    _, text = line.split("\t")
    all_documents.append(sent_tokenize(text))
    line = input_file.readline()
input_file.close()

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
for i in tqdm(range(len(all_documents))):
    for j in range(len(all_documents[i])):
        all_documents[i][j] = tokenizer.tokenize(all_documents[i][j])

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(args.max_predictions_per_seq,
                         max(1, int(round(len(tokens) * args.masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        output_tokens[index] = masked_token
        masked_lms.append((index, tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
    return output_tokens, masked_lm_positions, masked_lm_labels

rng = np.random.RandomState(args.seed)
tokenizer_vocab = list(tokenizer.vocab.keys())
train_input_ids = []
train_attention_masks = []
train_segment_ids = []
train_masked_lm_labels = []
train_next_sentence_labels = []
for _ in range(args.dupe_factor):
    for document_index in tqdm(range(len(all_documents))):
        document = all_documents[document_index]

        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = args.max_seq_length - 3

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if rng.random() < args.short_seq_prob:
            target_seq_length = rng.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = rng.randint(1, len(current_chunk))

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    # Random next
                    is_random_next = False
                    if len(current_chunk) == 1 or rng.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = rng.randint(0, len(random_document))
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    tokens = []
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)

                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                        tokens, args.masked_lm_prob, args.max_predictions_per_seq, tokenizer_vocab, rng)
                    
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    while len(input_ids) < args.max_seq_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                    assert len(input_ids) == args.max_seq_length
                    assert len(input_mask) == args.max_seq_length
                    assert len(segment_ids) == args.max_seq_length
                    next_sentence_label = 1 if is_random_next else 0
                    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
                    coverted_masked_lm_labels = [-100] * args.max_seq_length
                    for k, v in zip(masked_lm_positions, masked_lm_ids):
                        coverted_masked_lm_labels[k] = v

                    train_input_ids.append(input_ids)
                    train_attention_masks.append(input_mask)
                    train_segment_ids.append(segment_ids)
                    train_masked_lm_labels.append(coverted_masked_lm_labels)
                    train_next_sentence_labels.append(next_sentence_label)
                current_chunk = []
                current_length = 0
            i += 1

with open("cache/imdb_further_pretraining", "wb") as writer:
    pickle.dump((np.array(train_input_ids), np.array(train_attention_masks), np.array(train_segment_ids),
                 np.array(train_masked_lm_labels), np.array(train_next_sentence_labels)), writer)