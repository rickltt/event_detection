from torch.utils.data import Dataset
import json
import os
from tqdm import tqdm
from const import ACE_EVENTS, DUEE_EVENTS
import torch
import codecs

def valid_sequence_output(sequence_output, valid_mask, attention_mask):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask

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

class ED_Dataset(Dataset):
    def __init__(self, args, mode):
        fname = os.path.join(args.data_dir, '{}.json'.format(mode))
        fin = open(fname, 'r')
        data = json.load(fin)
        fin.close()
        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))

        self.max_seq_length = args.max_seq_length
        self.tokenizer = args.tokenizer

        self.label2id = args.label2id
        self.id2label = args.id2label

        pad_token_label_id = -100
        for instance in data_iterator:
            words = instance['tokens']
            events = instance['events']
            tokens = []
            valid_mask = []
            for word in words:
                word_tokens = self.tokenizer.tokenize(word)
                # Chinese may have space for separate, use unk_token instead
                if word_tokens == []:
                    word_tokens = [self.tokenizer.unk_token]
                for i, word_token in enumerate(word_tokens):
                    if i == 0:
                        valid_mask.append(1)
                    else:
                        valid_mask.append(0)
                    tokens.append(word_token)

            label_ids = ['O'] * len(words)
            for event in events:
                event_type = event['event_type']
                start = event['start']
                end = event['end']

                for idx in range(start,end):
                    if idx == start:
                        label_ids[idx] = 'B-{}'.format(event_type)
                    else:
                        label_ids[idx] = 'I-{}'.format(event_type)
            label_ids = [ self.label2id[i] for i in label_ids]
            special_tokens_count = 2
            if len(tokens) > self.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
                valid_mask = valid_mask[: (self.max_seq_length - special_tokens_count)]
            
            # add sep token
            tokens += ['[SEP]']
            label_ids += [pad_token_label_id]
            valid_mask.append(1)
            segment_ids = [0] * len(tokens)

            # add cls token
            tokens = ['[CLS]'] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [0] + segment_ids
            valid_mask.insert(0, 1)

            # input_ids, attention_mask
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # padding
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [0] * padding_length
            valid_mask += [0] * padding_length
            while (len(label_ids) < self.max_seq_length):
                label_ids.append(pad_token_label_id)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length
            assert len(valid_mask) == self.max_seq_length

            sample = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'valid_mask': valid_mask,
                'label_ids': label_ids
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

#         samples = []
#         print('docs:',len(lines))
#         for line in lines:
#             line = line.rstrip()
#             doc = json.loads(line)
#             # docids = doc["id"]
#             sentences = []
#             for sent in doc["content"]:
#                 sentences.append({
#                     'tokens': sent['tokens'],
#                     'events': []
#                     }) 
#             for event in doc["events"]:
#                 event_type = event['type']
#                 for mention in event["mention"]:
#                     start = mention["offset"][0]
#                     end = mention["offset"][1]
#                     sent_id = mention["sent_id"]
#                     trigger = mention["trigger_word"]
#                     assert trigger == ' '.join(sentences[sent_id]['tokens'][start:end])
#                     sentences[sent_id]["events"].append({
#                         'event_type': event_type,
#                         'trigger': trigger,
#                         'start': start,
#                         'end': end   
#                     })
#             for sent in sentences:
#                 sents = [i['tokens'] for i in samples]
#                 if sent['tokens'] not in sents:
#                     samples.append(sent)

#         self.instances = []
#         data_iterator = tqdm(samples, desc="Loading: {} Data".format(mode))
        
#         pad_token_label_id = -100
#         for item in data_iterator:
#             words = item['tokens']
#             events = item['events']
#             tokens = []
#             valid_mask = []
#             for word in words:
#                 word_tokens = self.tokenizer.tokenize(word)
#                 for i, word_token in enumerate(word_tokens):
#                     if i == 0:
#                         valid_mask.append(1)
#                     else:
#                         valid_mask.append(0)
#                     tokens.append(word_token)
    
#             label_ids = ['O'] * len(words)
#             for event in events:
#                 event_type = event['event_type']
#                 start = event['start']
#                 end = event['end']   
#                 for idx in range(start,end):
#                     if idx == start:
#                         label_ids[idx] = 'B-{}'.format(event_type)
#                     else:
#                         label_ids[idx] = 'I-{}'.format(event_type)
#             label_ids = [ self.label2id[i] for i in label_ids]
#             special_tokens_count = 2
#             if len(tokens) > self.max_seq_length - special_tokens_count:
#                 tokens = tokens[: (self.max_seq_length - special_tokens_count)]
#                 label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
#                 valid_mask = valid_mask[: (self.max_seq_length - special_tokens_count)]
            
#             # add sep token
#             tokens += ['[SEP]']
#             label_ids += [pad_token_label_id]
#             valid_mask.append(1)
#             segment_ids = [0] * len(tokens)

#             # add cls token
#             tokens = ['[CLS]'] + tokens
#             label_ids = [pad_token_label_id] + label_ids
#             segment_ids = [0] + segment_ids
#             valid_mask.insert(0, 1)

#             # input_ids, attention_mask
#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             input_mask = [1] * len(input_ids)

#             # padding
#             padding_length = self.max_seq_length - len(input_ids)
#             input_ids += [0] * padding_length
#             input_mask += [0] * padding_length
#             segment_ids += [0] * padding_length
#             valid_mask += [0] * padding_length
#             while (len(label_ids) < self.max_seq_length):
#                 label_ids.append(pad_token_label_id)

#             assert len(input_ids) == self.max_seq_length
#             assert len(input_mask) == self.max_seq_length
#             assert len(segment_ids) == self.max_seq_length
#             assert len(label_ids) == self.max_seq_length
#             assert len(valid_mask) == self.max_seq_length

#             instance = {
#                 'input_ids': input_ids,
#                 'attention_mask': input_mask,
#                 'token_type_ids': segment_ids,
#                 'valid_mask': valid_mask,
#                 'label_ids': label_ids
#             }
#             self.instances.append(instance)

#     def __getitem__(self, index):
#         return self.instances[index] 
#     def __len__(self):
#         return len(self.instances) 
    

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

