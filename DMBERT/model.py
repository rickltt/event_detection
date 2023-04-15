import torch
import torch.nn as nn
from transformers import BertModel

class DMBERT(nn.Module):

    def __init__(self, args):
        super(DMBERT, self).__init__()
        self.num_labels = args.num_labels
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.maxpooling = nn.MaxPool1d(args.max_seq_length)
        self.loss_fct = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None, maskL=None, maskR=None, labels=None, mode=None):
        batchSize=input_ids.size(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        conved=outputs[0]
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(batchSize,self.bert.config.hidden_size)
        pooledR=self.maxpooling(R).contiguous().view(batchSize,self.bert.config.hidden_size)
        pooled=torch.cat((pooledL,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
        pooled=self.dropout(pooled)
        logits=self.classifier(pooled)

        loss= self.loss_fct(logits, labels)

        if mode == 'train':
            return loss
        else:
            return logits, loss

