import os
import json
import torch
import codecs
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from const import *
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

class ED_Dataset(Dataset):
    def __init__(self, args, mode):
        fname = os.path.join(args.data_dir, '{}.json'.format(mode))
        fin = open(fname, 'r')
        data = json.load(fin)
        # random.shuffle(data)
        # data = data[:1000]
        fin.close()
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
        data_iterator = tqdm(instances, desc="Loading: {} Data".format(mode))

        self.max_seq_length = args.max_seq_length
        self.tokenizer = args.tokenizer
        self.label2id = args.label2id
        self.id2label = args.id2label

        self.samples = []
        for instance in data_iterator:
            tokens = instance['tokens']
            trigger = instance['trigger_tokens']
            start = instance['trigger_start']
            end = instance['trigger_end']
            assert trigger == ' '.join(tokens[start:end])
            event_type = instance['event_type']

            textL = self.tokenizer.tokenize(" ".join(tokens[:start]))
            textR = self.tokenizer.tokenize(" ".join(tokens[start:end]))+['[unused1]']+self.tokenizer.tokenize(" ".join(tokens[end:]))

            maskL = [1.0] * (len(textL) +1) + [0.0] * len(textR)
            maskR = [0.0] * (len(textL) +1) + [1.0] * len(textR)

            if len(maskL)>self.max_seq_length:
                maskL = maskL[:self.max_seq_length]
            if len(maskR)>self.max_seq_length:
                maskR = maskR[:self.max_seq_length]

            text = textL + ['[unused0]'] + textR
            input_ids = self.tokenizer.convert_tokens_to_ids(text)
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]        
            token_type_ids = [0] * len(input_ids)

            assert len(input_ids)==len(maskL)
            assert len(input_ids)==len(maskR)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            maskL = maskL + ([0.0] * padding_length)
            maskR = maskR + ([0.0] * padding_length)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(token_type_ids) == self.max_seq_length
            
            label = args.label2id[event_type]
            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'maskL': maskL,
                'maskR': maskR,
                'labels': label
            }
            self.samples.append(sample)

    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  

# class MavenDataset(Dataset):
#     def __init__(self, args, mode):
#         f = codecs.open(os.path.join(args.data_dir, "{}.jsonl".format(mode)), 'r', encoding="utf-8", errors="ignore") 
#         lines = f.readlines()
#         f.close()
#         self.max_seq_length = args.max_seq_length
#         self.tokenizer = args.tokenizer

#         self.label2id = args.label2id
#         self.id2label = args.id2label
#         print('docs:',len(lines))

#         sentences = []
#         for line in lines:
#             line = line.rstrip()
#             doc = json.loads(line)
#             for event in doc["events"]:
#                 event_type = event['type']
#                 for mention in event["mention"]:
#                     start = mention["offset"][0]
#                     end = mention["offset"][1]
#                     sent_id = mention["sent_id"]
#                     trigger = mention["trigger_word"]
#                     assert trigger == ' '.join(doc["content"][sent_id]['tokens'][start:end])
                        
#                     sentences.append({
#                         'tokens': doc["content"][sent_id]['tokens'],
#                         'event_type': event_type,
#                         'trigger': trigger,
#                         'start': start,
#                         'end': end   
#                     })
#             for nagtive in doc["negative_triggers"]:
#                 sentences.append({
#                     'tokens': doc["content"][nagtive['sent_id']]['tokens'],
#                     'event_type': 'None',
#                     'trigger': nagtive['trigger_word'],
#                     'start': nagtive['offset'][0],
#                     'end': nagtive['offset'][1] 
#                 })

#         self.samples = []
#         data_iterator = tqdm(sentences, desc="Loading: {} Data".format(mode))
        
#         for instance in data_iterator:
#             tokens = instance['tokens']
#             trigger = instance['trigger']
#             start = instance['start']
#             end = instance['end']
#             assert trigger == ' '.join(tokens[start:end])
#             event_type = instance['event_type']

#             textL = self.tokenizer.tokenize(" ".join(tokens[:start]))
#             textR = self.tokenizer.tokenize(" ".join(tokens[start:end]))+['[unused1]']+self.tokenizer.tokenize(" ".join(tokens[end:]))

#             maskL = [1.0] * (len(textL) +1) + [0.0] * len(textR)
#             maskR = [0.0] * (len(textL) +1) + [1.0] * len(textR)

#             if len(maskL)>self.max_seq_length:
#                 maskL = maskL[:self.max_seq_length]
#             if len(maskR)>self.max_seq_length:
#                 maskR = maskR[:self.max_seq_length]

#             text = textL + ['[unused0]'] + textR
#             input_ids = self.tokenizer.convert_tokens_to_ids(text)
#             if len(input_ids) > self.max_seq_length:
#                 input_ids = input_ids[:self.max_seq_length]        
#             token_type_ids = [0] * len(input_ids)

#             assert len(input_ids)==len(maskL)
#             assert len(input_ids)==len(maskR)

#             # The mask has 1 for real tokens and 0 for padding tokens. Only real
#             # tokens are attended to.
#             attention_mask = [1] * len(input_ids)
#             # Zero-pad up to the sequence length.
#             padding_length = self.max_seq_length - len(input_ids)
#             input_ids = input_ids + ([0] * padding_length)
#             attention_mask = attention_mask + ([0] * padding_length)
#             token_type_ids = token_type_ids + ([0] * padding_length)

#             maskL = maskL + ([0.0] * padding_length)
#             maskR = maskR + ([0.0] * padding_length)

#             assert len(input_ids) == self.max_seq_length
#             assert len(attention_mask) == self.max_seq_length
#             assert len(token_type_ids) == self.max_seq_length
            
#             label = args.label2id[event_type]
#             sample = {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'token_type_ids': token_type_ids,
#                 'maskL': maskL,
#                 'maskR': maskR,
#                 'labels': label
#             }
#             self.samples.append(sample)

#     def __getitem__(self, index):
#         return self.samples[index] 
#     def __len__(self):
#         return len(self.samples) 
     
# if __name__ == '__main__':
#     from arguments import get_args
#     args = get_args()
#     from transformers import AutoTokenizer
#     args.data_dir = '/home/tongtao.ling/ltt_code/event_detection/data/ace'
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     args.tokenizer = tokenizer
#     processor = DataProcessor(args)
#     args.label2id = processor.label2id
#     args.id2label = processor.id2label
#     args.num_labels = processor.num_labels

#     dataset = ED_Dataset(args, 'dev')



#     print(len(dataset))