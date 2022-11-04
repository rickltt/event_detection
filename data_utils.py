import os
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
import json
class MultiProcessor(DataProcessor):
    def __init__(self,labels):
        super().__init__()
        # self.labels = args.labels
        self.labels = labels
        self.labels.append("none")
        self.label2id = {i: j for j,i in enumerate(self.labels)}
    def get_examples(self, data_dir, split):
        examples = []
        path = os.path.join(data_dir, "{}.json".format(split))
        with open(path,'r') as f:
            instances = json.load(f)
            for instance in instances:
                guid = instance['guid']          
                text_a = instance['sentence']
                label = []
                event_list = []
                events = instance['events']

                if events == []:
                    label.append(self.label2id["none"])
                else:
                    for event in events: 
                        event_type = event['event_type']
                        trigger = event['trigger']

                        start = event['start']
                        end = event['end']  
                        event_list.append({
                            "trigger":trigger,
                            "event_type":event_type,
                            "start":start,
                            "end":end
                        })
                        label.append(self.label2id[event_type])
                multi_labels = [0]* len(self.labels)
                for i in label:
                    multi_labels[int(i)] = 1
                meta = {
                    'event_list': event_list
                }
                example = InputExample(guid=guid,text_a=text_a,meta=meta,label=multi_labels)
                examples.append(example)
        return examples


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