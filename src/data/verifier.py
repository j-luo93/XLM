"""
Verify the boundaries of BPE segments.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from arglib import init_g_attr
from devlib import get_length_mask, get_range, get_tensor, get_zeros

from ..model.graphormer import GraphData

Tensor = torch.Tensor


@dataclass
class GraphInfo:
    bpe_mask: Tensor
    word_mask: Tensor
    word_lengths: Tensor
    word2bpe: Tensor


@init_g_attr
class Verifier:

    def __init__(self, data_path, lgs, supervised_graph):
        super().__init__()
        src_lang, tgt_lang = lgs.split('-')
        if supervised_graph:
            src_loaded = torch.load(Path(data_path) / f'train.{src_lang}.grf.pth')
            tgt_loaded = torch.load(Path(data_path) / f'train.{tgt_lang}.grf.pth')
            self.src_graph = src_loaded['graph']
            self.tgt_graph = tgt_loaded['graph']
        else:
            src_loaded = torch.load(Path(data_path) / f'valid.{src_lang}.pth')
        self.dico = src_loaded['dico']
        self.incomplete_bpe = set()
        incomplete_idx = list()
        for bpe, idx in self.dico.word2id.items():
            if bpe.endswith('@@'):
                self.incomplete_bpe.add(bpe)
                incomplete_idx.append(idx)
        idx = get_zeros(len(self.dico)).bool()
        idx[incomplete_idx] = True
        self.incomplete_idx = idx

    def get_graph_info(self, data) -> GraphInfo:
        """
        bpe_mask is the mask that marks the boundaries the words.
        word_mask is the mask that marks whether or not a position is padded.
        word2bpe is the matrix that maps from word ids to bpe positions.
        """
        data = data.t()
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

    def get_graph_target(self, indices: List[int]) -> GraphData:
        # FIXME(j_luo)
        # node_features =
        # edge_index =
        # edge_norm =
        # edge_type =
        return GraphData(None, None, edge_norm, edge_type)

    def get_graph_loss(self, graph_data: GraphData, graph_target: GraphData) -> Tuple[Tensor, Tensor]:
        """
        Sizes for graph_data and graph_target:
                            graph_data      graph_target
            edge_norm:      E               E
            edge_type:      E x nr          E
        where E = bs x wl x wl.
        """
        # NOTE(j_luo) This determines whether it's an actual edge (in contrast to a padded edge) or not.
        edge_mask = graph_target.edge_norm

        loss_edge_type = graph_data.edge_type.clamp(
            min=1e-8).log().gather(1, graph_target.edge_type.view(-1, 1)).view(-1)
        loss_edge_type = (loss_edge_type * edge_mask).sum()

        loss_edge_norm = (graph_data.edge_norm.clamp(min=1e-8, max=1.0).log() * edge_mask).sum()

        return loss_edge_type, loss_edge_norm
