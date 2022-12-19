import os
import pickle

import numpy as np
import torch
import argparse
import logging
import youtokentome as yttm

from src.preprocessing import data_utils

from torchtext import transforms as T


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_tokenizer(data_path, vocab_size, tokenizer_path,
                    pad_id=0, unk_id=1, bos_id=2, eos_id=3):

    yttm.BPE.train(data=data_path, vocab_size=vocab_size, model=tokenizer_path,
                   pad_id=pad_id, unk_id=unk_id, bos_id=bos_id, eos_id=eos_id)


def tokenize(texts, tokenizer_path):
    tokenizer = yttm.BPE(model=tokenizer_path)
    tokenized = tokenizer.encode(texts, output_type=yttm.OutputType.ID, dropout_prob=0.1, bos=True, eos=True)
    return tokenized


def detokenize(tokens, tokenizer_path):
    tokenizer = yttm.BPE(model=tokenizer_path)
    texts = tokenizer.decode(tokens)
    return texts


def pad_sequences(sequences, max_len):
    if max_len == -1:
        max_len = np.max([len(seq) for seq in sequences])

    pad_func = T.Sequential(
        T.Truncate(max_seq_len=max_len),
        T.ToTensor(padding_value=data_utils.PAD_IDX),
    )

    def to_tensor(sequences):
        return torch.LongTensor(pad_func(sequences))

    return to_tensor(sequences)


def main(cfg):
    data = {
        "train": data_utils.read_parallel_corpus(cfg.train_src, cfg.train_trg, cfg.max_seq_len, cfg.lower_case),
        "val": data_utils.read_parallel_corpus(cfg.val_src, cfg.val_trg, None, cfg.lower_case),
        "test": data_utils.read_parallel_corpus(cfg.test_src, cfg.test_trg, None, cfg.lower_case),
    }

    for data_path, tokenizer_path, vocab_size in zip(
            (cfg.src_tokenizer_path, cfg.trg_tokenizer_path),
            (cfg.train_src, cfg.train_trg),
            (cfg.src_vocab_size, cfg.trg_vocab_size),
    ):
        if not os.path.exists(tokenizer_path):
            train_tokenizer(data_path=data_path, vocab_size=vocab_size, tokenizer_path=tokenizer_path,
                            pad_id=data_utils.PAD_IDX, unk_id=data_utils.UNK_IDX,
                            bos_id=data_utils.BOS_IDX, eos_id=data_utils.EOS_IDX)

    for split in data:

        data[split] = (
            pad_sequences(tokenize(data[split][0], cfg.src_tokenizer_path), cfg.max_seq_len),
            pad_sequences(tokenize(data[split][1], cfg.trg_tokenizer_path), cfg.max_seq_len),
        )
        path = cfg.save_data_dir + "/" + f"{split}.{cfg.src_lang}_{cfg.trg_lang}.pkl"

        with open(path, "wb") as file:
            pickle.dump({cfg.src_lang: data[split][0], cfg.trg_lang: data[split][0]}, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('--train-src', required=True, type=str, help='Path to training source data')
    parser.add_argument('--train-trg', required=True, type=str, help='Path to training target data')
    parser.add_argument('--val-src', required=True, type=str, help='Path to validation source data')
    parser.add_argument('--val-trg', required=True, type=str, help='Path to validation target data')
    parser.add_argument('--test-src', required=True, type=str, help='Path to test source data')
    parser.add_argument('--test-trg', required=True, type=str, help='Path to test target data')
    parser.add_argument('--src-lang', required=True, type=str, help='Source language abbreviation')
    parser.add_argument('--trg-lang', required=True, type=str, help='Target language abbreviation')
    parser.add_argument('--src-vocab-size', type=int, default=1000, help='Source vocabulary size')
    parser.add_argument('--trg-vocab-size', type=int, default=1000, help='Target vocabulary size')
    parser.add_argument('--max-seq-len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--lower-case', action='store_true')
    parser.add_argument('--save-data-dir', required=True, type=str, help='Output directory for the prepared data')
    parser.add_argument('--src-tokenizer-path', required=True, type=str, help='Path to source tokenizer model')
    parser.add_argument('--trg-tokenizer-path', required=True, type=str, help='Path to target tokenizer model')

    cfg = parser.parse_args()
    logger.info(str(cfg))
    main(cfg)

