from __future__ import print_function
import os
import sys
import time
import math
import argparse
import logging
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.preprocessing import data_utils
from src.models.transfomer import Transformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_checkpoint(model_path, load_optimizer=True, load_scheduler=True):
    model_state = torch.load(model_path)
    cfg = model_state["cfg"]

    model = Transformer(embedding_size=cfg.embedding_size,
                        src_vocab_size=cfg.src_vocab_size,
                        trg_vocab_size=cfg.trg_vocab_size,
                        src_pad_idx=data_utils.PAD_IDX,
                        num_heads=cfg.n_heads,
                        num_encoder_layers=cfg.n_layers,
                        num_decoder_layers=cfg.n_layers,
                        forward_expansion=4,
                        dropout=cfg.dropout,
                        max_len=cfg.max_seq_len,
                        use_cuda=cfg.use_cuda)

    model.load_state_dict(model_state['model_params'])
    return_list = [model_state, model]

    if load_optimizer:
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(model_state["optimizer_params"])
        return_list.append(optimizer)
        if load_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
            )
            scheduler.load_state_dict(model_state["scheduler_params"])
            return_list.append(scheduler)

    return return_list


def create_model(cfg):

    logger.info('Creating new model parameters..')
    model = Transformer(embedding_size=cfg.embedding_size,
                        src_vocab_size=cfg.src_vocab_size,
                        trg_vocab_size=cfg.trg_vocab_size,
                        src_pad_idx=data_utils.PAD_IDX,
                        num_heads=cfg.n_heads,
                        num_encoder_layers=cfg.n_layers,
                        num_decoder_layers=cfg.n_layers,
                        forward_expansion=4,
                        dropout=cfg.dropout,
                        max_len=cfg.max_seq_len,
                        use_cuda=cfg.use_cuda)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    model_state = {'cfg': cfg, 'curr_epochs': 0, 'train_steps': 0}

    return model_state, model, optimizer, scheduler


def main(cfg):
    logger.info('Loading training and development data..')
    train_dataloader, val_dataloader = data_utils.load_train_data(cfg.train_path, cfg.val_path,
                                                       cfg.src_lang, cfg.trg_lang, cfg.batch_size)

    if cfg.use_cuda and not torch.cuda.is_available():
        logger.warning("Cuda is not available. CPU will be used")
        cfg.use_cuda = False

    device = torch.device("cuda") if cfg.use_cuda else torch.device("cpu")
    
    # Create a new model or load an existing one.
    if os.path.exists(cfg.model_path):
        model_state, model, optimizer, scheduler = load_checkpoint(cfg.model_path,
                                                                   load_optimizer=True, load_scheduler=True)
    else:
        model_state, model, optimizer, scheduler = create_model(cfg)

    init_epoch = model_state['curr_epochs']
    if init_epoch >= cfg.max_epochs:
        logger.info(f'Training is already complete. Current_epoch:{init_epoch}, max_epoch:{cfg.max_epochs}')
        sys.exit(0)

    criterion = nn.CrossEntropyLoss(ignore_index=data_utils.PAD_IDX)

    for epoch in range(init_epoch + 1, cfg.max_epochs + 1):
        # Execute training steps for 1 epoch.
        logger.info("Training")
        start = time.time()
        train_loss, train_ppl = train(train_dataloader, model, criterion, optimizer, scheduler, model_state, device)
        logger.info(f"Epoch {epoch}. Loss: {train_loss}. Perplexity: {train_ppl}")
        logger.info(f"Training epoch took: {timedelta(seconds=int(time.time() - start))}")

        # Execute a validation step.
        logger.info("Validation")
        eval_loss, eval_ppl = eval(val_dataloader, model, criterion, device)
        logger.info(f'Epoch {epoch}. Loss: {eval_loss}. Perplexity: {eval_ppl}')

        # Save the model checkpoint in every 1 epoch.
        model_state['curr_epochs'] += 1
        model_state['model_params'] = model.state_dict()
        model_state["optimizer_params"] = optimizer.state_dict()
        model_state["scheduler_params"] = scheduler.state_dict()

        torch.save(model_state, cfg.model_path)
        logger.info('The model checkpoint file has been saved')


def train(train_dataloader, model, criterion, optimizer, scheduler, model_state, device):  # TODO: fix cfg
    model.train()
    train_loss, train_loss_total = 0.0, 0.0
    n_words, n_words_total = 0, 0
    n_sents, n_sents_total = 0, 0

    for batch_idx, batch in enumerate(train_dataloader):

        # if batch_idx > 10:
        #     break

        optimizer.zero_grad()
        inp_data = batch[0].T.to(device)
        target = batch[1].T.to(device)
        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        train_loss_total += float(loss.item())
        n_words_total += torch.sum((inp_data > 0).sum(axis=0) - 1)
        n_sents_total += inp_data.size(1)
        model_state['train_steps'] += 1

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx + 1:>5,} of {len(train_dataloader):>5,}. Loss: {loss.item():.7}")

        # Display training status
        if model_state['train_steps'] % cfg.display_freq == 0:
            loss_int = (train_loss_total - train_loss)
            n_words_int = (n_words_total - n_words)

            loss_per_words = loss_int / n_words_int
            avg_ppl = math.exp(loss_per_words) if loss_per_words < 300 else float("inf")

            logger.info(f"Epoch {model_state['curr_epochs']:<3}. " +
                        f"Step {model_state['train_steps']:<10}. " +
                        f"Perplexity {avg_ppl:<10.2f}")
            train_loss, n_words, n_sents = (train_loss_total, n_words_total, n_sents_total)

    mean_loss = np.mean(train_loss_total) / n_sents_total
    scheduler.step(mean_loss)
    return mean_loss, math.exp(train_loss_total / n_words_total)


def eval(val_dataloader, model, criterion, device):
    model.eval()
    eval_loss_total = 0.0
    n_words_total, n_sents_total = 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):

            # if batch_idx > 10:
            #     break

            inp_data = batch[0].T.to(device)
            target = batch[1].T.to(device)

            output = model(inp_data, target[:-1, :])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            eval_loss_total += float(loss.item())

            n_words_total += torch.sum((inp_data > 0).sum(axis=0) - 1)
            n_sents_total += inp_data.size(1)

    mean_loss = eval_loss_total / n_sents_total
    return mean_loss, math.exp(eval_loss_total / n_words_total)


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
    parser.add_argument('--display-freq', type=int, default=10)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--use-cuda', type=bool, default=False)

    cfg = parser.parse_args()
    logger.info(str(cfg))
    main(cfg)
