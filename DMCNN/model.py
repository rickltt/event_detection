import torch
import torch.nn as nn


class DMCNN(nn.Module):

    def __init__(self, args):
        super(DMCNN, self).__init__()

        self.num_labels = args.num_labels
        self.embedding_dim = args.hidden_size
        self.embedding = nn.Embedding(len(args.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(args.vec_mat))
        self.embedding.weight.requires_grad = True

        self.cnn = nn.Conv1d(in_channels=args.hidden_size,
                            out_channels=args.hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1)

        self.fc = nn.Linear(args.hidden_size*2, self.num_labels)
        # self.fc=nn.Sequential(nn.Linear(args.hidden_size*2, 256),
        #               nn.Dropout(0.5),
        #               nn.Linear(256, self.num_labels))
        
        self.dropout = nn.Dropout(args.dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.maxpooling = nn.MaxPool1d(128)

    def forward(self, input_ids, maskL, maskR, labels, mode):

        encoder_out=self.embedding(input_ids)
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out=self.cnn(encoder_out)
        encoder_out= encoder_out.permute(0, 2, 1)

        # encoder_out=encoder_out[0]
        batchSize=encoder_out.shape[0]
        conved = encoder_out
        conved = conved.transpose(1, 2)
        conved = conved.transpose(0, 1)
        L = (conved * maskL).transpose(0, 1)
        R = (conved * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(batchSize, self.embedding_dim)
        pooledR = self.maxpooling(R).contiguous().view(batchSize, self.embedding_dim)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        loss = self.loss_fct(logits, labels)
        if mode == 'train':
            return loss
        else:
            return logits, loss
