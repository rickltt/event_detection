import torch
import random
import numpy as np
from data_utils import MultiProcessor,calc_metric
from constant import cnc_labels, maven_labels
from openprompt.prompts import SoftVerbalizer,ManualTemplate,MixedTemplate, SoftTemplate
from openprompt.plms import load_plm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from openprompt import PromptForClassification,PromptDataLoader
from transformers import AdamW
from tqdm import tqdm
from arguments import get_args_parser
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, prompt_model, train_dataloader, eval_dataloader):

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=1e-8)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    criterion = torch.nn.BCEWithLogitsLoss()
    max_val_p = 0.0
    max_val_r = 0.0
    max_val_f1 = 0.0

    for epoch in range(args.num_train_epochs):
        logger.info("Epoch:{}".format(epoch+1))
        tot_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, inputs in enumerate(epoch_iterator):
            inputs = inputs.to(args.device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = criterion(logits, labels.float())
            # loss = criterion(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            # if step % 100 == 1:
            #     logger.info("epoch:{},average_loss:{}".format(epoch+1,tot_loss/(step+1)))

        val_p, val_r, val_f1 = evaluate(args, prompt_model, eval_dataloader)

        print('>> val_p: {:.4f}, val_r: {:.4f}, val_f1: {:.4f}'.format(val_p, val_r, val_f1))
        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
            if not os.path.exists('output'):
                os.mkdir('output')
            path = 'output/{}_{}_val_f1_{}'.format(args.prompt, args.dataset, round(val_f1, 4))
            torch.save(prompt_model.state_dict(), path)
            print('>> saved model: {}'.format(path)) 
        if val_p > max_val_p:
            max_val_p = val_p
        if val_r > max_val_r:
            max_val_r = val_r

    return path

def evaluate(args, prompt_model, eval_dataloader):
    true_set = []
    pred_set = []
    prompt_model.eval()
    with torch.no_grad():
        for inputs in tqdm(eval_dataloader, desc='Evaluating'):
            guids = inputs['guid']
            inputs = inputs.to(args.device)
            logits = prompt_model(inputs)
            labels = inputs['label']

            true = labels.detach().cpu().numpy()
            pred = torch.sigmoid(logits).ge(0.5).int().detach().cpu().numpy()

            for i,j in enumerate(true):
                if j[-1] == 1:
                    continue
                for k,v in enumerate(j):
                    if v == 1:
                        true_set.append((guids[i],args.labels[k]))

            for i,j in enumerate(pred):
                if j[-1] == 1:
                    continue
                for k,v in enumerate(j):
                    if v == 1 :
                        pred_set.append((guids[i],args.labels[k]))

    p,r,f1 = calc_metric(true_set,pred_set)

    return p, r, f1

def main():

    args = get_args_parser()

    if args.overwrite_output_dir:
        os.system('rm -rf output/')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device

    plm, tokenizer, _, WrapperClass = load_plm(args.model_type, args.model_name_or_path)

    dataset = args.dataset

    dataset_files = {
        'maven': './data/maven/' ,
        'cnc':'./data/cnc'
    }

    labels_map = {
        'maven': maven_labels ,
        'cnc': cnc_labels
    }

    args.labels = labels_map[dataset]

    dataset_path = dataset_files[dataset]


    processor = MultiProcessor(args.labels)
    train_dataset = processor.get_train_examples(dataset_path)
    dev_dataset = processor.get_dev_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)

    dataset = {}
    dataset['train'], dataset['dev'], dataset['test'] = train_dataset, dev_dataset, test_dataset

    logger.info("***** dataset:%s *****", args.dataset)
    logger.info("***** model:%s *****", args.model_type)
    logger.info("***** Size of train_dataset:%d *****", len(dataset['train']))
    logger.info("***** Size of dev_dataset:%d *****", len(dataset['dev']))
    logger.info("***** Size of test_dataset:%d *****", len(dataset['test']))
    logger.info("***** Prompt:%s *****", args.prompt)
    # prompt
    # prompt1: test_p: 0.9434, test_r: 0.8600, test_f1: 0.8998 cnc
    # prompt2: test_p: 0.9406, test_r: 0.8453, test_f1: 0.8904 cnc
    # prompt3: test_p: 0.9509, test_r: 0.8564, test_f1: 0.9012 cnc
    # prompt4: test_p: 0.9615, test_r: 0.8729, test_f1: 0.9151 cnc

    if args.prompt == 'p1':
        prompt = 'What happened? {"placeholder":"text_a"} This sentence decribes a {"mask"} event'
    elif args.prompt == 'p2':
        prompt = '{"placeholder":"text_a"} What event does the previous sentence describe? It was a {"mask"} event'
    elif args.prompt == 'p3':  
        prompt = '{"placeholder":"text_a"} It was {"mask"}'
    elif args.prompt == 'p4':
        prompt = 'A {"mask"} event: {"placeholder":"text_a"}'
    elif args.prompt == 'soft':
        prompt = '{"soft"} {"soft"} {"placeholder":"text_a"} {"soft"} {"soft"} {"mask"}.'

    if args.prompt == 'soft':
        mytemplate = SoftTemplate(plm,tokenizer,prompt)
        # mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=prompt)
    else:
        mytemplate = ManualTemplate(text = prompt, tokenizer = tokenizer)

    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(args.labels))

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length,batch_size=args.batch_size,shuffle=True)

    dev_dataloader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, batch_size=args.batch_size,shuffle=False)

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, batch_size=args.batch_size,shuffle=False)

    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)

    prompt_model=  prompt_model.to(args.device)

    best_model_path = train(args, prompt_model, train_dataloader, dev_dataloader)

    prompt_model.load_state_dict(torch.load(best_model_path))

    test_p,test_r,test_f1 = evaluate(args, prompt_model, test_dataloader)

    result = 'test_p: {:.4f}, test_r: {:.4f}, test_f1: {:.4f}, dataset: {}, prompt: {}'.format(test_p, test_r, test_f1, args.dataset, args.prompt)

    logger.info(result)
    
    with open('./result.txt','a') as f:
        f.write(result + '\n')

if __name__ == '__main__':
    main()