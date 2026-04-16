import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import AutoTokenizer
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small', use_fast=False)
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        data = []
        for nl in nl_lines:
            prompted_nl = "generate SQL: " + nl
            input_ids = tokenizer(prompted_nl)["input_ids"]
            data.append({
                "input_ids": input_ids,
                "decoder_start_token_id": self.decoder_start_token_id,
            })

        if split == "test":
            return data

        sql_path = os.path.join(data_folder, f"{split}.sql")
        sql_lines = load_lines(sql_path)

        for example, sql in zip(data, sql_lines):
            target_ids = tokenizer(sql)["input_ids"]
            example["target_ids"] = target_ids

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    decoder_targets = [torch.tensor(example["target_ids"], dtype=torch.long) for example in batch]
    initial_decoder_inputs = [
        torch.tensor([example["decoder_start_token_id"]], dtype=torch.long) for example in batch
    ]
    decoder_inputs = [
        torch.cat([initial_decoder_input, decoder_target[:-1]], dim=0)
        for initial_decoder_input, decoder_target in zip(initial_decoder_inputs, decoder_targets)
    ]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    initial_decoder_inputs = [
        torch.tensor([example["decoder_start_token_id"]], dtype=torch.long) for example in batch
    ]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x

def print_data_statistics(data_folder):
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small', use_fast=False)

    before = {}
    after = {}
    for split in ["train", "dev"]:
        nl_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        sql_lines = load_lines(os.path.join(data_folder, f"{split}.sql"))
        raw_input_ids = [tokenizer(line)["input_ids"] for line in nl_lines]
        raw_target_ids = [tokenizer(line)["input_ids"] for line in sql_lines]

        dataset = T5Dataset(data_folder, split)
        processed_input_ids = [example["input_ids"] for example in dataset.data]
        processed_target_ids = [example["target_ids"] for example in dataset.data]

        raw_input_lengths = [len(input_ids) for input_ids in raw_input_ids]
        raw_target_lengths = [len(target_ids) for target_ids in raw_target_ids]
        raw_input_vocab = set(token_id for input_ids in raw_input_ids for token_id in input_ids)
        raw_target_vocab = set(token_id for target_ids in raw_target_ids for token_id in target_ids)
        before[split] = {
            "num_examples": len(raw_input_ids),
            "mean_input_len": sum(raw_input_lengths) / len(raw_input_lengths),
            "mean_target_len": sum(raw_target_lengths) / len(raw_target_lengths),
            "input_vocab_size": len(raw_input_vocab),
            "target_vocab_size": len(raw_target_vocab),
        }

        processed_input_lengths = [len(input_ids) for input_ids in processed_input_ids]
        processed_target_lengths = [len(target_ids) for target_ids in processed_target_ids]
        processed_input_vocab = set(token_id for input_ids in processed_input_ids for token_id in input_ids)
        processed_target_vocab = set(token_id for target_ids in processed_target_ids for token_id in target_ids)
        after[split] = {
            "num_examples": len(processed_input_ids),
            "mean_input_len": sum(processed_input_lengths) / len(processed_input_lengths),
            "mean_target_len": sum(processed_target_lengths) / len(processed_target_lengths),
            "input_vocab_size": len(processed_input_vocab),
            "target_vocab_size": len(processed_target_vocab),
        }

    print("Table 1: Data statistics before preprocessing")
    print(f"{'Statistics Name':35s} {'Train':>10s} {'Dev':>10s}")
    print(f"{'Number of examples':35s} {before['train']['num_examples']:10d} {before['dev']['num_examples']:10d}")
    print(f"{'Mean sentence length':35s} {before['train']['mean_input_len']:10.2f} {before['dev']['mean_input_len']:10.2f}")
    print(f"{'Mean SQL query length':35s} {before['train']['mean_target_len']:10.2f} {before['dev']['mean_target_len']:10.2f}")
    print(f"{'Vocabulary size (natural language)':35s} {before['train']['input_vocab_size']:10d} {before['dev']['input_vocab_size']:10d}")
    print(f"{'Vocabulary size (SQL)':35s} {before['train']['target_vocab_size']:10d} {before['dev']['target_vocab_size']:10d}")

    print("\nTable 2: Data statistics after preprocessing")
    print("Model name: google-t5/t5-small")
    print(f"{'Statistics Name':35s} {'Train':>10s} {'Dev':>10s}")
    print(f"{'Mean sentence length':35s} {after['train']['mean_input_len']:10.2f} {after['dev']['mean_input_len']:10.2f}")
    print(f"{'Mean SQL query length':35s} {after['train']['mean_target_len']:10.2f} {after['dev']['mean_target_len']:10.2f}")
    print(f"{'Vocabulary size (natural language)':35s} {after['train']['input_vocab_size']:10d} {after['dev']['input_vocab_size']:10d}")
    print(f"{'Vocabulary size (SQL)':35s} {after['train']['target_vocab_size']:10d} {after['dev']['target_vocab_size']:10d}")

if __name__ == "__main__":
    print_data_statistics("data")
