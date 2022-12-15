from __future__ import print_function
import os
import pickle
import sys
import time
import math
import argparse
import logging
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.optim as optim

from src.preprocessing import data_utils
from src.preprocessing.data_utils import load_train_data
from src.models.transfomer import Transformer

use_cuda = torch.cuda.is_available()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_model(cfg):
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info('Creating new model parameters..')
    model = Transformer(cfg.embedding_size, cfg.src_vocab_size, cfg.trg_vocab_size,
                        data_utils.PAD, cfg.n_heads, cfg.n_layers,
                        cfg.n_layers, 4, cfg.dropout, cfg.max_seq_len, device)
    model_state = {'cfg': cfg, 'curr_epochs': 0, 'train_steps': 0}

    # If cfg.model_path exists, load model parameters.
    if os.path.exists(cfg.model_path):
        logger.info('Reloading model parameters..')
        model_state = torch.load(cfg.model_path)
        model.load_state_dict(model_state['model_params'])

    return model, model_state


def main(cfg):
    logger.info('Loading training and development data..')
    train_dataloader, val_dataloader = load_train_data(cfg.train_path, cfg.val_path,
                                                       cfg.src_lang, cfg.trg_lang, cfg.batch_size)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    # Create a new model or load an existing one.
    model, model_state = create_model(cfg)
    init_epoch = model_state['curr_epochs']
    if init_epoch >= cfg.max_epochs:
        logger.info(f'Training is already complete. Current_epoch:{init_epoch}, max_epoch:{cfg.max_epochs}')
        sys.exit(0)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=data_utils.PAD)

    if cfg.log:
        log_train_file = cfg.log + '.train.log'
        log_dev_file = cfg.log + '.valid.log'
        if not os.path.exists(log_train_file) and os.path.exists(log_dev_file):
            with open(log_train_file, 'w') as log_tf, open(log_dev_file, 'w') as log_df:
                log_tf.write('epoch,ppl,sents_seen\n')
                log_df.write('epoch,ppl,sents_seen\n')
        logger.info('Training and validation log will be written in {} and {}'
              .format(log_train_file, log_dev_file))

    for epoch in range(init_epoch + 1, cfg.max_epochs + 1):
        # Execute training steps for 1 epoch.
        train_loss = train(train_dataloader, model, criterion, optimizer, scheduler, model_state, device)
        logger.info(f'Epoch {epoch} Train_ppl: {train_loss:.2f}')

        # Execute a validation step.
        # eval_loss, eval_sents = eval(model, criterion, val_dataloader)
        # logger.info(f'Epoch {epoch} Eval_ppl: {eval_loss:.2f} Sents seen: {eval_sents}')

        # # Save the model checkpoint in every 1 epoch.
        # model_state['curr_epochs'] += 1
        # model_state['model_params'] = model.state_dict()
        # torch.save(model_state, cfg.model_path)
        # logger.info('The model checkpoint file has been saved')
        #
        # if cfg.log and log_train_file and log_dev_file:
        #     with open(log_train_file, 'a') as log_tf, open(log_dev_file, 'a') as log_df:
        #         log_tf.write('{epoch},{ppl:0.2f},{sents}\n'.format(
        #             epoch=epoch, ppl=train_loss, sents=train_sents, ))
        #         log_df.write('{epoch},{ppl:0.2f},{sents}\n'.format(
        #             epoch=epoch, ppl=eval_loss, sents=eval_sents, ))


def train(train_dataloader, model, criterion, optimizer, scheduler, model_state, device):  # TODO: fix cfg
    model.train()
    train_losses = []
    batch_losses = []
    start = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        inp_data = batch[0].T.to(device)
        target = batch[1].T.to(device)
        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        batch_losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx + 1:>5,} of {len(train_dataloader):>5,}. "
                        f"Time: {timedelta(seconds=int(time.time() - start)):}. Loss: {loss.item():.7}")

    mean_loss = np.sum(batch_losses) / len(train_dataloader)
    scheduler.step(mean_loss)
    train_losses.append(mean_loss)

    logger.info(f"Mean loss: {mean_loss}")
    logger.info(f"Training epoch took: {timedelta(seconds=int(time.time() - start))}")
    return train_losses



def eval(model, criterion, dev_iter):
    model.eval()
    eval_loss_total = 0.0
    n_words_total, n_sents_total = 0, 0

    logger.info('Evaluation')
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_iter):
            enc_inputs, enc_inputs_len = batch.src
            dec_, dec_inputs_len = batch.trg
            dec_inputs = dec_[:, :-1]
            dec_targets = dec_[:, 1:]
            dec_inputs_len = dec_inputs_len - 1

            dec_logits, *_ = model(enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len)
            step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))
            eval_loss_total += float(step_loss.data[0])
            n_words_total += torch.sum(dec_inputs_len)
            n_sents_total += dec_inputs_len.size(0)
            logger.info('  {} samples seen'.format(n_sents_total))

    # return per_word_loss
    return math.exp(eval_loss_total / n_words_total), n_sents_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparams')
    # data loading params
    parser.add_argument('--train-path', required=True, help='Path to the preprocessed training data')
    parser.add_argument('--val-path', required=True, help='Path to the preprocessed validation data')
    parser.add_argument('--src-lang', required=True, type=str, help='Source language abbreviation')
    parser.add_argument('--trg-lang', required=True, type=str, help='Target language abbreviation')


    # network params
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)

    # training params
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--src-vocab-size', type=int, default=1000)
    parser.add_argument('--trg-vocab-size', type=int, default=1000)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--max_grad-norm', type=float, default=None)
    parser.add_argument('--n-warmup-steps', type=int, default=4000)
    parser.add_argument('--display-freq', type=int, default=100)
    parser.add_argument('--log', default=None)
    parser.add_argument('--model-path', type=str, required=True)

    cfg = parser.parse_args()
    logger.info(str(cfg))
    main(cfg)
