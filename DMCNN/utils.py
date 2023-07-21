import os
import json
import numpy as np
import torch
from const import *
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score

def calculate_scores(preds, labels, dimE):
    positive_labels = list(range(1,dimE)) #assume 0 is NA
    pre = precision_score(labels, preds, labels=positive_labels, average='micro')
    recall = recall_score(labels, preds, labels=positive_labels, average='micro')
    f1 = f1_score(labels, preds, labels=positive_labels, average='micro')
    return pre, recall, f1

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
    # if args.dataset == 'duee':
    #     lines = lines[1:]
    #     unk_embedding = np.random.randn(300)
    #     unk_embedding = unk_embedding.astype(str)
    #     unk_embedding = '<UNK> ' + ' '.join(unk_embedding)
    #     lines.insert(0,unk_embedding)

    #     pad_embedding = np.random.randn(300)
    #     pad_embedding = pad_embedding.astype(str)
    #     pad_embedding = '<PAD> ' + ' '.join(pad_embedding)
    #     lines.insert(0,pad_embedding)

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
        labels = ['None']
        if 'ace' in args.dataset:
            events = ACE_EVENTS
        elif 'ere' in args.dataset:
            events = ERE_EVENTS
        elif 'maven' in args.dataset:
            events = MAVEN_EVENTS
        else:
            raise ValueError("incorrect dataset!")

        for label in events:
            labels.append(label)
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
        self.embedding_dict, self.word2id, self.vec_mat = args.embedding_dict, args.word2id, args.vec_mat

        instances = []
        for i in data:
            tokens = i['tokens']
            events = i['events']
            labels = ['None'] * len(tokens)
            for event in events:
                event_type = event["event_type"]
                for idx in range(event['start'],event['end']):
                    labels[idx] = event_type
            for j,token in enumerate(tokens):
                instances.append({
                    'tokens':tokens,
                    'trigger_tokens': token,
                    'trigger_start':j,
                    'trigger_end':j+1,
                    'event_type': labels[j]
                })

        self.samples = []
        data_iterator = tqdm(instances, desc="Loading: {} Data".format(mode))

        for instance in data_iterator:
            tokens = instance['tokens']
            trigger = instance['trigger_tokens']
            start = instance['trigger_start']
            end = instance['trigger_end']
            assert trigger == ' '.join(tokens[start:end])
            event_type = instance['event_type']

            textL = tokens[:start]
            textR = tokens[start:]
            maskL = [1.0] * len(textL) + [0.0] * len(textR)
            maskR = [0.0] * len(textL) + [1.0] * len(textR)

            input_ids = []
            for w in tokens:
                if w in self.word2id:
                    input_ids.append(self.word2id[w])
                else:
                    input_ids.append(self.word2id['<UNK>'])

            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                maskL = maskL[:self.max_seq_length]
                maskR = maskR[:self.max_seq_length]        
            else:
                padding_length = self.max_seq_length - len(input_ids)
                input_ids += ([0] * padding_length)
                maskL += ([0.0] * padding_length)
                maskR += ([0.0] * padding_length)

            assert len(input_ids)==len(maskL)
            assert len(input_ids)==len(maskR)
            assert len(input_ids) == self.max_seq_length

            label = args.label2id[event_type]
            sample = {
                'input_ids': input_ids,
                'maskL': maskL,
                'maskR': maskR,
                'labels': label
            }
            self.samples.append(sample)

    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  

