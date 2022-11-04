import argparse

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Prompt-based text classification.")
    
    parser.add_argument(
        "--dataset",
        default='maven',
        type=str,
        help="The input data directory."
    )

    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Select the model type selected to be used from bert, roberta, albert, now only support roberta."
    )

    parser.add_argument(
        "--model_name_or_path",
        default='/home/ltt/ltt_code/bert/bert-large-uncased',
        type=str,
        help="Path to pretrained model or shortcut name of the model."
    )

    parser.add_argument(
        "--prompt",
        default='p1',
        type=str,
        help="template function."
    )

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform."
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    # parser.add_argument(
    #     "--overwrite_output_dir", type=bool, default=True, help="Overwrite the content of the output directory."
    # )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory."
    )

    args = parser.parse_args()

    return args