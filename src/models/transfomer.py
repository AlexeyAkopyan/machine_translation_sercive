from __future__ import print_function

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size,
                 src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers,
                 forward_expansion, dropout, max_len, use_cuda):
        super(Transformer, self).__init__()
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size).to(self.device)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size).to(self.device)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size).to(self.device)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size).to(self.device)

        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers,
                                          forward_expansion, dropout).to(self.device)

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device))

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)

        out = self.fc_out(out)
        return out