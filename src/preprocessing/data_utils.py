# -*- coding: utf-8 -*-
import logging
import pickle

from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler

logger = logging.getLogger(__name__)


PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


def read_parallel_corpus(src_path, tgt_path, max_len, lower_case=False):
    logger.info(f"Reading examples from {src_path} and {tgt_path}")
    src_sents, tgt_sents = [], []
    empty_lines, exceed_lines = 0, 0
    with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
        for idx, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
            if src_line.strip() == '' or tgt_line.strip() == '':  # remove empty lines
                empty_lines += 1
                continue
            if lower_case:  # check lower_case
                src_line = src_line.lower()
                tgt_line = tgt_line.lower()

            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if max_len is not None and (len(src_line.split()) > max_len or len(tgt_line.split()) > max_len):
                exceed_lines += 1
                continue
            src_sents.append(src_line)
            tgt_sents.append(tgt_line)

    logger.info(f'Filtered {empty_lines} empty lines and {exceed_lines} lines exceeding the length {max_len}')
    logger.info(f'Result: {len(src_sents)} lines remained')
    return src_sents, tgt_sents


def load_train_data(train_path, val_path, src_lang, trg_lang, batch_size):
    with open(train_path, "rb") as file:
        train_data = pickle.load(file)

    with open(val_path, "rb") as file:
        val_data = pickle.load(file)

    train_data = TensorDataset(train_data[src_lang], train_data[trg_lang])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_data[src_lang], val_data[trg_lang])
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return train_dataloader, val_dataloader


