from utils import ED_Dataset, collate_fn, DataProcessor, calc_metric
import torch
from model import ED_BERT
import numpy as np
import random
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import argparse

from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

import logging
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, model, train_dataset, dev_dataset):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size 
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps

    # print('logging_steps:',args.logging_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Dataset = %s", args.dataset)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_dev_f1 = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()

            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "mask_position":batch[3],
                "sep_position":batch[4],
                "label":batch[5],
                "mode": "train"
            }

            loss = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        result, _ = evaluate(args, model, dev_dataset)
                        if best_dev_f1 < result['f1']:
                            best_dev_f1 = result['f1']
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            args.tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            # logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset):

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    trues = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch.values())
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "mask_position":batch[3],
                "sep_position":batch[4],
                "label": batch[5],
                "mode": "test"
            }
            # tmp_eval_loss, logits = model(**inputs)
            tmp_eval_loss, pred = model(**inputs)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        labels = inputs['label']
        # logits = F.softmax(logits,dim=1)

        preds.extend(pred)
        trues.extend(labels.detach().cpu().tolist())


    eval_loss = eval_loss / nb_eval_steps

    trues_list, preds_list = [], []

    for idx, (i,j) in enumerate(zip(preds, trues)):
        if j[0] != 1:
            for k in range(args.num_labels):
                if j[k] == 1:
                    trues_list.append((idx, args.id2label[k]))
        if i != []:
            for t in i:
                preds_list.append((idx, args.id2label[t]))

    p, r, f1 = calc_metric(trues_list, preds_list)
    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[trigger classification]\tP={:.4f}\tR={:.4f}\tF1={:.4f}\n'.format(p, r, f1)

    print(metric)

    result = {}
    result['eval_loss'] = eval_loss
    result['p'] = p
    result['r'] = r
    result['f1'] = f1
    return result, metric


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--dataset', default='ace++', type=str, help='ace, ace++, ere, maven')
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout_prob.")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=str, default='0.5',
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=98,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datafiles = {
        'ace': '../data/ace',
        'ace++': '../data/ace++',
        'ere': '../data/ere',
        'maven': '../data/maven'
    }

    args.data_dir = datafiles[args.dataset]
    # Set seed
    set_seed(args)
    
    processor = DataProcessor(args)
    args.label2id = processor.label2id
    args.id2label = processor.id2label
    args.num_labels = processor.num_labels

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    num_added_tokens = tokenizer.add_tokens(processor.event_tokens)
    args.tokenizer = tokenizer

    args.event_ids = tokenizer.convert_tokens_to_ids(processor.event_tokens)

    train_dataset = ED_Dataset(args, processor, 'train')
    dev_dataset = ED_Dataset(args, processor, 'dev')
    test_dataset = ED_Dataset(args, processor, 'test')

    # bert = AutoModel.from_pretrained(args.model_name_or_path)

    model = ED_BERT(args)
    args.model_type = 'ED_PRC'

    # state_dict = torch.load(os.path.join(args.mrc_model_path, "model"))
    model.bert.resize_token_embeddings(len(tokenizer)) 

    # bert_state_dict = {}
    # for k,v in state_dict.items():
    #     if k in ['fc.weight','fc.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias']:
    #         continue
    #     new_key = k.replace('bert.','')
    #     bert_state_dict[new_key] = v
    # model.bert.load_state_dict(bert_state_dict)

    model.to(args.device)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
        logger.info("Saving model checkpoint to %s", output_dir)

    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)
        model.to(args.device)

        _, metric = evaluate(args, model, test_dataset)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('***** Model: {} Predict in {} test dataset *****'.format(args.model_type, args.dataset))
            writer.write("{} \n".format(metric))
        
if __name__ == '__main__':
    main()