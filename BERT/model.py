import torch
import torch.nn as nn
from transformers import BertModel
from utils import valid_sequence_output
from crf import CRF

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.num_labels = args.num_labels
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.loss_type = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            valid_mask,
            label_ids,
            mode
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.contiguous().view(-1)[active_loss]

        loss = self.loss_type(active_logits, active_labels)
        if mode == 'train':
            return loss
        else:
            return logits, loss
    

class BertCrf(nn.Module):
    def __init__(self, args):
        super(BertCrf, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.num_labels = args.num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            valid_mask,
            label_ids,
            mode
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        
        labels = torch.where(label_ids >= 0, label_ids, torch.zeros_like(label_ids))
        loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
        if mode == 'train':
            return loss
        else:
            tags = self.crf.decode(logits, attention_mask)
            return tags, loss

