from __future__ import print_function
import torch.nn as nn
import logging

from src.modules.embedding import PositionalEncoding

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size,
                 src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers,
                 forward_expansion, dropout, max_len, device):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size).to(device)
        self.src_position_embedding = PositionalEncoding(embedding_size, dropout).to(device)

        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size).to(device)
        self.trg_position_embedding = PositionalEncoding(embedding_size, dropout).to(device)

        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers,
                                          forward_expansion, dropout).to(device)

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size).to(device)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        embed_src = self.src_position_embedding(self.src_word_embedding(src))
        embed_trg = self.trg_position_embedding(self.trg_word_embedding(trg))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)

        out = self.fc_out(out)
        return out