import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import os
import argparse
from utils import DataProcessor, ED_Dataset, collate_fn, calculate_scores
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import DMCNN
from tqdm import tqdm, trange

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
    

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    best_f1 = 0.0
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator):
            model.zero_grad()
            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "input_ids":batch[0],
                "maskL":batch[1],
                "maskR":batch[2],
                "labels":batch[3],
                "mode": "train"
            }
            loss = model(**inputs)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            optimizer.step()
        result, _ = evaluate(args, model, dev_dataset)  
        if best_f1 < result['f1']:
            best_f1 = result['f1']
            output_dir = os.path.join(args.output_dir, "best_checkpoint")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)


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
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch.values())
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "maskL":batch[1],
                "maskR":batch[2],
                "labels":batch[3],
                "mode": "test"
            }
            logits, tmp_eval_loss = model(**inputs)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    precision, recall, f1 = calculate_scores(preds, out_label_ids, args.num_labels)

    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[trigger classification]\tP={:.4f}\tR={:.4f}\tF1={:.4f}\n'.format(precision, recall, f1)

    print(metric)
    results = {}
    results['eval_loss'] = eval_loss
    results['p'] = precision
    results['r'] = recall
    results['f1'] = f1
    return results, metric

def main():
    # Hyper Parameters

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--dataset', default='ace', type=str, help='ace, ace++, duee')
    parser.add_argument("--embedding_file", default='../data/100.utf8', type=str)
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout_prob.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--hidden_size", type=int, default=100, help="hidden size for word2vec")
    parser.add_argument("--rnn_hidden", type=int, default=300, help="hidden size for rnn")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers for rnn")

    args = parser.parse_args()

    set_seed(args)

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
        'maven': '../data/maven',
    }
    
    args.data_dir = datafiles[args.dataset]
    # if 'ace' in args.dataset:
    #     args.embedding_file = '../data/100.utf8'
    #     args.hidden_size = 100
    # else:
    #     args.embedding_file = '../data/token_vec_300.bin'
    #     args.hidden_size = 300

    processor = DataProcessor(args)
    args.label2id = processor.label2id
    args.id2label = processor.id2label
    args.num_labels = processor.num_labels
    args.embedding_dict, args.word2id, args.vec_mat = processor.embedding_dict, processor.word2id, processor.vec_mat
    
    train_dataset = ED_Dataset(args, 'train')
    dev_dataset = ED_Dataset(args, 'dev')
    test_dataset = ED_Dataset(args, 'test')

    model = DMCNN(args)
    args.model_type = 'DMCNN'
    model.to(args.device)

    # Training
    if args.do_train:
        train(args, model, train_dataset, dev_dataset)

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
        checkpoint = os.path.join(args.output_dir, 'last_checkpoint')
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