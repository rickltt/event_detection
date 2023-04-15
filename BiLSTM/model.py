# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from crf import CRF

class BiLSTM(nn.Module):

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.embedding_dim = args.hidden_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        # self.fc=nn.Sequential(nn.Linear(args.rnn_hidden, 128),
        #             nn.Dropout(args.dropout_prob),
        #             nn.Linear(128, self.num_labels))
        self.fc = nn.Linear(args.rnn_hidden, args.num_labels)
        self.dropout = nn.Dropout(args.dropout_prob)

        self.lstm = nn.LSTM(args.hidden_size, args.rnn_hidden//2, args.num_layers,
                            bidirectional=True, batch_first=True)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, input_ids, input_mask = None, label_ids = None, mode = None):
        # mask = input_mask  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out=self.embedding(input_ids)

        encoder_out, _ = self.lstm(encoder_out)
        encoder_out = self.dropout(encoder_out)
        logits=self.fc(encoder_out)

        active_logits = logits.contiguous().view(-1, self.num_labels)
        active_labels = label_ids.contiguous().view(-1)

        loss = self.loss_fn(active_logits, active_labels)   
        if mode == 'train':
            return loss
        else:
            return logits, loss
        

class BiLSTMCRF(nn.Module):

    def __init__(self, args):
        super(BiLSTMCRF, self).__init__()

        self.embedding_dim = args.hidden_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc = nn.Linear(args.rnn_hidden, args.num_labels)

        self.lstm = nn.LSTM(args.hidden_size, args.rnn_hidden//2, args.num_layers,
                            bidirectional=True, batch_first=True)
        
        self.crf = CRF(num_tags=args.num_labels, batch_first=True)

    def forward(self, input_ids, input_mask, label_ids, mode):
        mask = input_mask  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out=self.embedding(input_ids)

        encoder_out, _ = self.lstm(encoder_out)
        encoder_out = self.dropout(encoder_out)
        logits=self.fc(encoder_out)

        labels = torch.where(label_ids >= 0, label_ids, torch.zeros_like(label_ids))
        loss = self.crf(emissions=logits, tags=labels, mask=mask)
        loss = -1 * loss
        
        if mode == 'train':
            return loss
        else:
            logits = self.crf.decode(logits, mask)
            return logits, loss

