from torch.utils.data import Dataset
import json
import os
from tqdm import tqdm
from const import ACE_EVENTS, DUEE_EVENTS
import torch
import random

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
        else:
            events = DUEE_EVENTS
        
        event_tokens = []
        event_tokens = [ '<' + e.lower() + '>' for e in events]
        event_tokens.insert(0, '<none>')
        self.event_tokens = event_tokens

        for e in events:
            labels.append(e)
        self.labels = labels
        self.id2label = {i:j for i,j in enumerate(labels) }
        self.label2id = {j:i for i,j in enumerate(labels) }
        self.num_labels = len(labels)

class ED_Dataset(Dataset):
    def __init__(self, args, processor, mode):
        fname = os.path.join(args.data_dir, '{}.json'.format(mode))
        fin = open(fname, 'r')
        data = json.load(fin)
        fin.close()
        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))

        tokenizer = args.tokenizer
        label2id = processor.label2id
        max_seq_length = args.max_seq_length
        event_tokens = processor.event_tokens
        for instance in data_iterator:
            words = instance['tokens']
            events = instance['events']

            tokens = []
            for word in words:
                word_tokens = tokenizer.tokenize(word)
                # Chinese may have space for separate, use unk_token instead
                if word_tokens == []:
                    word_tokens = [self.tokenizer.unk_token]
                for word_token in word_tokens:
                    tokens.append(word_token)

            label = [0] * len(label2id)
            if events == []:
                label[0] = 1
            else:
                for e in events:
                    event_type = e['event_type']
                    label[label2id[event_type]] = 1

            prompt = 'it was a [MASK] event.'
            special_tokens_count = 4 + len(tokenizer.tokenize(prompt)) + len(event_tokens)
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]

            text =  ['[CLS]'] + tokens + ['[SEP]'] + tokenizer.tokenize(prompt) + ['[SEP]'] + event_tokens + ['[SEP]']

            input_ids = tokenizer.convert_tokens_to_ids(text)
            input_masks = [1] * len(input_ids)
            mask_position = text.index('[MASK]')
            sep_position = text.index('[SEP]') + len(tokenizer.tokenize(prompt)) + 1

            assert input_ids[sep_position] == 102

            padding_length = max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            input_masks += [0] * padding_length
            segment_ids = [0] * len(input_ids)
           
            assert mask_position == tokenizer.convert_ids_to_tokens(input_ids).index('[MASK]')
            assert len(input_ids) == args.max_seq_length
            assert len(input_masks) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length


            item = {
                'input_ids': input_ids,
                'attention_mask': input_masks,
                'token_type_ids': segment_ids,
                'mask_position': mask_position,
                'sep_position': sep_position +1 ,
                'label': label,
            } 
            self.samples.append(item)

    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  


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

    print('proposed: {} \t correct: {} \t gold: {}'.format(num_proposed, num_correct, num_gold))

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
