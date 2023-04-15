import os
import json
import numpy as np
import torch
from const import ACE_EVENTS, DUEE_EVENTS
from torch.utils.data import Dataset
from tqdm import tqdm

def collate_fn(batch):
    new_batch = { key: [] for key in batch[0].keys()}
    for b in batch:
        for key in new_batch:
            new_batch[key].append(b[key]) 
    for b in new_batch:
        new_batch[b] = torch.tensor(new_batch[b], dtype=torch.long)
    return new_batch

def load_embedding_dict(args):
    with open(args.embedding_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.dataset == 'duee':
        lines = lines[1:]
        unk_embedding = np.random.randn(300)
        unk_embedding = unk_embedding.astype(str)
        unk_embedding = '<UNK> ' + ' '.join(unk_embedding)
        lines.insert(0,unk_embedding)

        pad_embedding = np.random.randn(300)
        pad_embedding = pad_embedding.astype(str)
        pad_embedding = '<PAD> ' + ' '.join(pad_embedding)
        lines.insert(0,pad_embedding)

    embedding_dict = {}
    for line in lines:
        split = line.split(" ")
        embedding_dict[split[0]] = np.array(list(map(float, split[1:])))
    word2id = {}
    for i,j in enumerate(embedding_dict.keys()):
        word2id[j] = i
    vec_mat = [i for i in embedding_dict.values()]
    vec_mat = np.array(vec_mat)
    
    return embedding_dict, word2id, vec_mat

class DataProcessor(object):
    def __init__(self, args):
        labels = ['O']
        if 'ace' in args.dataset:
            events = ACE_EVENTS
        else:
            events = DUEE_EVENTS

        for label in events:
            labels.append('B-{}'.format(label))
            labels.append('I-{}'.format(label))
        self.id2label = {i:j for i,j in enumerate(labels) }
        self.label2id = {j:i for i,j in enumerate(labels) }
        self.num_labels = len(labels)
        self.embedding_dict, self.word2id, self.vec_mat = load_embedding_dict(args)

class ED_Dataset(Dataset):
    def __init__(self, args, mode):
        fname = os.path.join(args.data_dir, '{}.json'.format(mode))
        fin = open(fname, 'r')
        data = json.load(fin)
        fin.close()
        self.max_seq_length = args.max_seq_length

        self.label2id = args.label2id
        self.id2label = args.id2label
    
        self.embedding_dict, self.word2id, self.vec_mat = args.embedding_dict, args.word2id, args.vec_mat

        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))

        pad_token_label_id = -100
        for instance in data_iterator:
            words = instance['tokens']
            events = instance['events']
            token_ids = []
            for word in words:
                if word in self.word2id:
                    token_ids.append(self.word2id[word])
                else:
                    token_ids.append(self.word2id['<UNK>']) 
            label_ids = ['O'] * len(words)
            for event in events:
                event_type = event['event_type']
                trigger = event['text']
                start = event['start']
                end = event['end']
                # assert ''.join(words[start:end]) == trigger
                for idx in range(start,end):
                    if idx == start:
                        label_ids[idx] = 'B-{}'.format(event_type)
                    else:
                        label_ids[idx] = 'I-{}'.format(event_type)
            label_ids = [ self.label2id[i] for i in label_ids]

            # padding
            seq_len = len(token_ids)
            padding_length = self.max_seq_length - seq_len
            if seq_len < self.max_seq_length:
                token_ids += ([0] * padding_length)
                input_mask = [1] * seq_len + [0] * padding_length
            else:
                token_ids = token_ids[:self.max_seq_length]
                input_mask = [1] * self.max_seq_length

        
            if len(label_ids) < self.max_seq_length:
                label_ids += ([pad_token_label_id] * (self.max_seq_length - len(label_ids)))
            else:
                label_ids = label_ids[:self.max_seq_length]

            assert len(token_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length
            sample = {
                'input_ids': token_ids,
                'input_mask': input_mask,
                'label_ids': label_ids
            }
            self.samples.append(sample)
    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  

def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result_trigger = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result_trigger.append([i, i + 1, '-'.join(labels[i][1:])])

    for item in result_trigger:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I' and item[2] == '-'.join(labels[j][1:]):
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result_trigger]

def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

