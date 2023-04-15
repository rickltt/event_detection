import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import os
from utils import ED_Dataset, DataProcessor, collate_fn, find_triggers, calc_metric
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import BiLSTM, BiLSTMCRF
import logging
from tqdm import tqdm, trange

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

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": args.learning_rate},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_proportion != 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Random seed = %d", args.seed)

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
                "input_mask":batch[1],
                "label_ids":batch[2],
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
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            result, _ = evaluate(args, model, dev_dataset)
            if best_dev_f1 < result['f1']:
                best_dev_f1 = result['f1']
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)


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
    preds = None
    trues = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch.values())
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "input_mask":batch[1],
                "label_ids":batch[2],
                "mode": "test"
            }
            pred, tmp_eval_loss = model(**inputs)
            if not args.use_crf:
                pred = pred.argmax(-1)
            label = inputs['label_ids']
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if preds is None:
            preds = pred.detach().cpu().numpy()
            trues = label.detach().cpu().numpy()
        else:
            preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
            trues = np.append(trues, label.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    print('eval_loss={}'.format(eval_loss))

    trues_list = [[] for _ in range(trues.shape[0])]
    preds_list = [[] for _ in range(preds.shape[0])]

    pad_token_label_id = - 100
    for i in range(trues.shape[0]):
        for j in range(trues.shape[1]):
            if trues[i, j] != pad_token_label_id:
                trues_list[i].append(args.id2label[trues[i][j]])
                preds_list[i].append(args.id2label[preds[i][j]])

    triggers_true = []
    triggers_pred = []
    for i, (label, pred) in enumerate(zip(trues_list, preds_list)):
        triggers_true_ = find_triggers(label)
        triggers_pred_ = find_triggers(pred)
        triggers_true.extend([(i, *item) for item in triggers_true_])
        triggers_pred.extend([(i, *item) for item in triggers_pred_])

    # print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    # print('P={:.4f}\tR={:.4f}\tF1={:.4f}'.format(trigger_p, trigger_r, trigger_f1))

    # print('[trigger identification]')
    # triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    # triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    # trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    # print('P={:.4f}\tR={:.4f}\tF1={:.4f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[trigger classification]\tP={:.4f}\tR={:.4f}\tF1={:.4f}\n'.format(trigger_p, trigger_r, trigger_f1)
    # metric += '[trigger identification]\tP={:.4f}\tR={:.4f}\tF1={:.4f}\n\n'.format(trigger_p_, trigger_r_, trigger_f1_)

    print(metric)

    result = {}
    result['eval_loss'] = eval_loss
    result['p'] = trigger_p
    result['r'] = trigger_r
    result['f1'] = trigger_f1
    return result, metric



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--dataset', default='duee', type=str, help='ace, ace++, duee')
    parser.add_argument("--embedding_file", default='../data/token_vec_300.bin', type=str)
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_crf", action="store_true",
                        help="Whether to use crf.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument(
        "--evaluate_after_epoch",
        action="store_true",
        help="Whether to run evaluation after every epoch.",
    )
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float,
                        help="dropout_prob.")
    parser.add_argument("--weight_decay", default=1e-8, type=float,
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
    parser.add_argument("--warmup_proportion", default=0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=str, default='1.0',
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument("--hidden_size", type=int, default=100, help="hidden size for word2vec")
    parser.add_argument("--rnn_hidden", type=int, default=256, help="hidden size for rnn")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers for rnn")

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
        'ace++': '../data/ace++',
        'ace': '../data/ace',
        'duee': '../data/duee1.0'
    }

    args.data_dir = datafiles[args.dataset]
    # Set seed
    set_seed(args)
    
    processor = DataProcessor(args)
    args.label2id = processor.label2id
    args.id2label = processor.id2label
    args.num_labels = processor.num_labels
    args.embedding_dict, args.word2id, args.vec_mat = processor.embedding_dict, processor.word2id, processor.vec_mat
    
    train_dataset = ED_Dataset(args, 'train')
    dev_dataset = ED_Dataset(args, 'dev')
    test_dataset = ED_Dataset(args, 'test')

    if args.use_crf:
        model = BiLSTMCRF(args)
        args.model_type = 'BiLSTMCRF'
    else:
        model = BiLSTM(args)
        args.model_type = 'BiLSTM'
    
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
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)
        model.to(args.device)
        if args.dataset == 'maven':
            _, metric = evaluate(args, model, dev_dataset)
        else:
            _, metric = evaluate(args, model, test_dataset)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('***** Model: {} Predict in {} test dataset *****'.format(args.model_type, args.dataset))
            writer.write("{} \n".format(metric))
            
if __name__ == '__main__':
    main()