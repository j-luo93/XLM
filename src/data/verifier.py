"""
Verify the boundaries of BPE segments.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from arglib import init_g_attr
from devlib import get_length_mask, get_range, get_tensor, get_zeros

Tensor = torch.Tensor


@dataclass
class GraphInfo:
    bpe_mask: Tensor
    word_mask: Tensor
    word_lengths: Tensor
    word2bpe: Tensor


@init_g_attr
class Verifier(nn.Module):

    def __init__(self, data_path, lgs):
        super().__init__()
        lang = lgs.split('-')[0]
        dico = torch.load(Path(data_path) / f'valid.{lang}.pth')['dico']
        self.incomplete_bpe = set()
        incomplete_idx = list()
        for bpe, idx in dico.word2id.items():
            if bpe.endswith('@@'):
                self.incomplete_bpe.add(bpe)
                incomplete_idx.append(idx)
        idx = get_zeros(len(dico)).bool()
        idx[incomplete_idx] = True
        self.register_buffer('incomplete_idx', idx)
        self.dico = dico

    def forward(self, data):
        """
        bpe_mask is the mask that marks the boundaries the words.
        word_mask is the mask that marks whether or not a position is padded.
        word2bpe is the matrix that maps from word ids to bpe positions.
        """
        data = get_tensor(data.t())  # NOTE(j_luo) Move to GPU if necessary. And remember to transpose it.
        bs, l = data.shape

        data_off_by_one = torch.cat([get_zeros(bs, 1).long(), data[:, :-1]], dim=1)
        # A new word is started if both the current bpe and the previous bpe are complete, and it's not a padding.
        new_word = ~self.incomplete_idx[data] & ~self.incomplete_idx[data_off_by_one] & (data != self.dico.pad_index)
        # Form distinct word ids by counting how many new words are formed up to now.
        word_ids = new_word.long().cumsum(dim=1)
        # bpe_mask: value is set to True iff both bpes belong to the same word.
        bpe_mask = (word_ids.unsqueeze(dim=1) == word_ids.unsqueeze(dim=2))

        # word_mask
        word_lengths, _ = word_ids.max(dim=1)
        max_len = max(word_lengths)
        word_mask = get_length_mask(word_lengths, max_len)  # size: bs x wl

        # word2bpe is computed by getting all the row and column indices correctly.
        word_idx = (word_ids - 1)  # size: bs x l
        bpe_idx = get_range(l, 2, 1)  # size: 1 x l
        batch_i = get_range(bs, 2, 0)  # size: bs x 1
        word2bpe = get_zeros(bs, max_len, l)
        word2bpe[batch_i, word_idx, bpe_idx] = 1.0

        return GraphInfo(bpe_mask, word_mask, word_lengths, word2bpe)
