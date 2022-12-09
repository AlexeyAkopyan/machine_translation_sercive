# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import pickle
import torch
import argparse
import logging
import sys

from src.preprocessing import data_utils
from src.preprocessing.data_utils import read_parallel_corpus
from src.preprocessing.data_utils import build_vocab
from src.preprocessing.data_utils import convert_text2idx

import torchtext
from torchtext import transforms as T


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


def train_tokenizer(data, vocab_size, prefix):
    torchtext.data.functional.generate_sp_model(
        data,
        vocab_size=vocab_size,
        model_type='unigram',
        model_prefix=prefix)


def tokenize(texts, path):
    model = torchtext.data.functional.load_sp_model(path)
    tokenizer = torchtext.data.functional.sentencepiece_numericalizer(model)
    tokens = list(tokenizer(texts))
    return tokens


def get_text_transformer(max_len):
    return T.Sequential(
        T.Truncate(max_len - 2),
        T.AddToken(token=BOS_IDX, begin=True),
        T.AddToken(token=EOS_IDX, begin=False),
        T.ToTensor(padding_value=PAD_IDX),
    )


def main(cfg):
    data = {
        "train": read_parallel_corpus(cfg.train_src, cfg.train_trg, cfg.max_len, cfg.lower_case),
        "val": read_parallel_corpus(cfg.val_src, cfg.val_trg, None, cfg.lower_case),
        "test": read_parallel_corpus(cfg.test_src, cfg.test_trg, None, cfg.lower_case),
    }

    text_transformer = get_text_transformer(max_len=cfg.max_len)

    if not os.path.exists(cfg.src_model_path):
        l = len(cfg.src_model_path) - 6 if cfg.src_model_path.endswith(".model") else len(cfg.src_model_path)
        train_tokenizer(cfg.train_src, cfg.src_size, prefix=cfg.src_model_path[:l])

    if not os.path.exists(cfg.trg_model_path):
        l = len(cfg.trg_model_path) - 6 if cfg.trg_model_path.endswith(".model") else len(cfg.trg_model_path)
        train_tokenizer(cfg.train_trg, cfg.trg_size, prefix=cfg.trg_model_path[:l])

    for split in data:

        data[split] = (
            torch.LongTensor(text_transformer(tokenize(data[split][0], cfg.src_model_path))),
            torch.LongTensor(text_transformer(tokenize(data[split][1], cfg.trg_model_path))),
        )
        path = cfg.save_data_dir + "/" + f"{split}.{cfg.src_lang}_{cfg.trg_lang}.pkl"

        with open(path, "wb") as file:
            pickle.dump({cfg.src_lang: data[split][0], cfg.trg_lang: data[split][0]}, file)

    # for split_name, split in train
    # torch.save(src_encoded, "train_" + )
    # torch.save(trg_encoded, 'en_encoded_tensor.pt')
    #
    # if cfg.vocab:
    #     src_counter, src_word2idx, src_idx2word = torch.load(cfg.vocab)['src_dict']
    #     tgt_counter, tgt_word2idx, tgt_idx2word = torch.load(cfg.vocab)['tgt_dict']
    # else:
    #     if cfg.share_vocab:
    #         logger.info('Building shared vocabulary')
    #         vocab_size = min(cfg.src_vocab_size, cfg.tgt_vocab_size) \
    #             if (cfg.src_vocab_size is not None and cfg.tgt_vocab_size is not None) else None
    #         counter, word2idx, idx2word = build_vocab(train_src + train_tgt, vocab_size,
    #                                                   cfg.min_word_count, data_utils.extra_tokens)
    #         src_counter, src_word2idx, src_idx2word = (counter, word2idx, idx2word)
    #         tgt_counter, tgt_word2idx, tgt_idx2word = (counter, word2idx, idx2word)
    #     else:
    #         src_counter, src_word2idx, src_idx2word = build_vocab(train_src, cfg.src_vocab_size,
    #                                                               cfg.min_word_count, data_utils.extra_tokens)
    #         tgt_counter, tgt_word2idx, tgt_idx2word = build_vocab(train_tgt, cfg.tgt_vocab_size,
    #                                                               cfg.min_word_count, data_utils.extra_tokens)
    # train_src, train_tgt = \
    #     convert_text2idx(train_src, src_word2idx), convert_text2idx(train_tgt, tgt_word2idx)
    # dev_src, dev_tgt = \
    #     convert_text2idx(dev_src, src_word2idx), convert_text2idx(dev_tgt, tgt_word2idx)
    #
    #
    # # Save source/target vocabulary and train/dev data
    # with open('{}.dict'.format(cfg.save_data), "wb") as file:
    #     pickle.dump(
    #         {
    #             'src_dict': (src_counter, src_word2idx, src_idx2word),
    #             'tgt_dict': (tgt_counter, tgt_word2idx, tgt_idx2word),
    #             'src_path': cfg.train_src,
    #             'tgt_path': cfg.train_tgt,
    #             'lower_case': cfg.lower_case
    #         }, file)
    #
    # with open('{}-train.t7'.format(cfg.save_data), "wb") as file:
    #     pickle.dump(
    #         {
    #             'train_src': train_src,
    #             'train_tgt': train_tgt,
    #             'dev_src': dev_src,
    #             'dev_tgt': dev_tgt,
    #             'src_dict': src_word2idx,
    #             'tgt_dict': tgt_word2idx,
    #         }, file)
    #
    # logger.info('Saved the vocabulary at {}.dict'.format(cfg.save_data))
    # logger.info('Saved the preprocessed train/dev data at {}-train.t7'.format(cfg.save_data))


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
    parser.add_argument('--src-size', type=int, default=1000, help='Source vocabulary size')
    parser.add_argument('--trg-size', type=int, default=1000, help='Target vocabulary size')
    # parser.add_argument('--min-word-count', type=int, default=1)
    parser.add_argument('--max-len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--lower-case', action='store_true')
    parser.add_argument('--share-vocab', action='store_true')
    parser.add_argument('--save-data-dir', required=True, type=str, help='Output directory for the prepared data')
    parser.add_argument('--src-model-path', required=True, type=str, help='Path to source tokenizer model')
    parser.add_argument('--trg-model-path', required=True, type=str, help='Path to target tokenizer model')


    cfg = parser.parse_args()
    logger.info(str(cfg))
    main(cfg)
