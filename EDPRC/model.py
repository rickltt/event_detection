import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
import torch.nn.functional as F
class Loss_fun(nn.Module):
    def __init__(self):
        super(Loss_fun, self).__init__()

    def forward(self, logit, target):
        log_softmax = nn.LogSoftmax(dim=0)
        loss = 0.0
        # loss_negative = 0.0
        # loss_positive = 0.0
        for i,j in zip(logit,target):
            # lo = 0.0
            if j[0]:
                loss_negative = - log_softmax(i)[0]
                lo = loss_negative
            else:
                j[0] = 1
                pos = torch.nonzero(j).squeeze(-1)
                pos_logit = torch.index_select(i,0,pos)
                pos_log_soft = - log_softmax(pos_logit)[1:]
                loss_positive = pos_log_soft.mean()
                j[0] = 0
                neg_logit = i[j==0]
                loss_negative = - log_softmax(neg_logit)[0]
                lo = loss_positive + loss_negative
            loss += lo
        loss /= logit.shape[0]
        return loss
    
class ED_BERT(nn.Module):
    def __init__(self, args):
        super(ED_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path, add_pooling_layer=False)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, len(args.tokenizer))
        self.fc = nn.Linear(self.bert.config.hidden_size, 2) 
        self.dropout = nn.Dropout(args.dropout_prob)
        self.event_ids = args.event_ids
        self.loss_fun = Loss_fun()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, token_type_ids, mask_position, sep_position, label, mode):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids) 
        sequence_output = bert_outputs[0]  # bs x seq x hidden_size

        event_tokens_state = []
        event_tokens_output = self.dropout(sequence_output)

        for i,j in zip(event_tokens_output,sep_position):
            event_out = i[j:j+len(self.event_ids),:]
            event_tokens_state.append(event_out)

        event_tokens_state = torch.stack(event_tokens_state)  # bs x event_types x hidden_size
        logits = self.fc(event_tokens_state) # bs x event_types x 2
        # loss_fun = nn.CrossEntropyLoss()
        #loss_span = loss_fun(logits.view(-1,2), label.view(-1))

        prediction_scores = self.mlm_head(sequence_output) # bs x seq x vocab_size

        masked_lm_loss = self.ce(prediction_scores.view(-1, self.bert.config.vocab_size), input_ids.view(-1))

        prediction_scores = self.dropout(prediction_scores)
        
        sequence_mask_output = prediction_scores[torch.arange(prediction_scores.size(0)), mask_position] # bs x vocab_size
        
        event_output = sequence_mask_output[:, self.event_ids] # bs x  event_types
        loss = self.loss_fun(event_output, label) + masked_lm_loss + self.ce(logits.view(-1,2), label.view(-1))
        # loss_fun = self.loss_fun
        # scores = F.softmax(event_output,dim=1)

        # preds = []
        # for score in scores:
        #     pred = torch.where(score > score[-1], 1.0, 0.0)
        #     if not pred.any():
        #         pred[-1] = 1.0
        #     preds.append(pred)
        # preds = torch.stack(preds)

        # event_output = outputs[:,:,self.event_ids]
        if mode == 'train':
            return loss
        else:
            preds = []
            span_pred = logits.argmax(-1) # bs x num_labels
            event_output = F.softmax(event_output,dim=1) # bs x num_labels
            for i,j in zip(span_pred, event_output):
                pred = []
                none_score = j[0]
                for idx, k in enumerate(i):
                    if k == 1 and j[idx] > none_score:
                        pred.append(idx)
                preds.append(pred)  

            return loss, preds