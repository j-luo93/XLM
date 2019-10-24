"""
Verify the boundaries of BPE segments.
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Tuple

import numpy as np
import torch

from arglib import add_argument, init_g_attr
from devlib import get_length_mask, get_range, get_tensor, get_zeros
from trainlib import Metric

from ..model.graphormer import GraphData

Tensor = torch.Tensor


@dataclass
class GraphInfo:
    bpe_mask: Tensor
    word_mask: Tensor
    word_lengths: Tensor
    word2bpe: Tensor


@unique
class EdgeType(Enum):
    # NOTE(j_luo) I'm now keeping three types of features: agent, theme and modifier.
    AGENT = 0
    THEME = 1
    MODIFIER = 2


@dataclass
class Edge:
    u: int
    v: int
    t: EdgeType


@dataclass
class Graph:
    edges: List[Edge]

    def __len__(self):
        return len(self.edges)


def _read_graphs(path):
    graphs = list()
    with path.open('r', encoding='utf8') as fin:
        for line in fin:
            segs = line.strip().split()
            edges = list()
            for seg in segs:
                u_idx, v_idx, t = seg.split('-')
                u_idx, v_idx = map(int, [u_idx, v_idx])
                if t not in {'agent', 'theme'}:
                    t = 'modifier'
                t = getattr(EdgeType, t.upper())
                e = Edge(u_idx, v_idx, t)
                edges.append(e)
            graphs.append(Graph(edges))
    return graphs


@init_g_attr(default='property')
class Verifier:

    add_argument('ae_noise_graph_mode', dtype=str, default='keep', choices=['keep', 'change'],
                 msg='determines how we handle the graph when noised is added to AE and graph supervision is used')

    def __init__(self, data_path, lgs, supervised_graph, ae_noise_graph_mode: 'n', ae_add_noise):
        super().__init__()
        src_lang, tgt_lang = lgs.split('-')
        if supervised_graph:
            self.graphs = dict()
            for lang in [src_lang, tgt_lang]:
                self.graphs[lang] = _read_graphs(data_path / f'train.{lang}.tok.cvtx.neo.txt')
        self.dico = torch.load(data_path / f'valid.{src_lang}.pth')['dico']
        self.incomplete_bpe = set()
        incomplete_idx = list()
        for bpe, idx in self.dico.word2id.items():
            if bpe.endswith('@@'):
                self.incomplete_bpe.add(bpe)
                incomplete_idx.append(idx)
        idx = get_zeros(len(self.dico)).bool()
        idx[incomplete_idx] = True
        self.incomplete_idx = idx

        self.ae_noise_graph_mode = ae_noise_graph_mode

    def get_graph_info(self, data) -> GraphInfo:
        """
        bpe_mask is the mask that marks the boundaries the words.
        word_mask is the mask that marks whether or not a position is padded.
        word2bpe is the matrix that maps from word ids to bpe positions.
        """
        data = data.t()
        bs, l = data.shape

        data_off_by_one = torch.cat([get_zeros(bs, 1).long(), data[:, :-1]], dim=1)
        # A new word is started if the previous bpe is complete and it's not a padding or <s>.
        new_word = ~self.incomplete_idx[data_off_by_one] & (data != self.dico.pad_index) & (data != self.dico.bos_index)
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

    def get_graph_target(
            self,
            data: Tensor,
            lang: str,
            max_len: int,
            indices: List[int],
            permutations: List[np.ndarray] = None,
            keep: np.ndarray = None) -> GraphData:
        # NOTE(j_luo)  If for some reason the first one is <s> or </s>, we need to offset the indices.
        if self.ae_noise_graph_mode == 'change':
            assert permutations is not None and keep is not None

        offsets = ((data[0] == self.dico.eos_index) | (data[0] == self.dico.bos_index)).long()
        graphs = [self.graphs[lang][i] for i in indices]
        bs = len(graphs)
        if len(offsets) != bs:
            raise RuntimeError('Something is terribly wrong.')

        ijkv = list()
        for batch_i, graph in enumerate(graphs):
            assert len(graph) <= max_len
            offset = offsets[batch_i].item()
            # Repeat the permutation and dropout processes and change the graph accordingly.
            if self.ae_noise_graph_mode == 'change':
                perm = permutations[batch_i].argsort()
                perm = np.arange(len(perm))[perm]
            for e in graph.edges:
                u = e.u + offset
                v = e.v + offset
                if self.ae_noise_graph_mode == 'change':
                    u = perm[e.u]
                    v = perm[e.v]
                    if keep[u, batch_i] and keep[v, batch_i]:
                        ijkv.append((batch_i, u, v, e.t.value))
                else:
                    ijkv.append((batch_i, u, v, e.t.value))

        i, j, k, v = zip(*ijkv)
        v = get_tensor(v)
        edge_norm = get_zeros([bs, max_len, max_len])
        edge_type = get_zeros([bs, max_len, max_len]).long()
        # NOTE(j_luo) Edges are symmetric.
        edge_norm[i, j, k] = 1.0
        edge_norm[i, k, j] = 1.0
        edge_type[i, j, k] = v
        edge_type[i, k, j] = v
        edge_norm = edge_norm.view(-1)
        edge_type = edge_type.view(-1)
        return GraphData(None, None, edge_norm, edge_type)

    def get_graph_loss(self, graph_data: GraphData, graph_target: GraphData, lang: str) -> Tuple[Metric, Metric]:
        """
            Sizes for graph_data and graph_target:
                                graph_data      graph_target
                edge_norm:      E               E
                edge_type:      E x nr          E
            where E = bs x wl x wl.
            """
        # NOTE(j_luo) This determines whether it's an actual edge (in contrast to a padded edge) or not.
        assert len(graph_data.edge_norm) == len(graph_target.edge_norm)
        assert len(graph_data.edge_type) == len(graph_target.edge_type)

        edge_mask = graph_target.edge_norm

        edge_types_log_probs = (1e-8 + graph_data.edge_type).log()
        loss_edge_type = -edge_types_log_probs.gather(1, graph_target.edge_type.view(-1, 1)).view(-1)
        loss_edge_type = (loss_edge_type * edge_mask).sum()

        log_edge_norm = (graph_data.edge_norm + 1e-8).clamp(max=1.0).log()
        loss_edge_norm = -(log_edge_norm * edge_mask).sum()

        weight = edge_mask.sum()
        loss_edge_type = Metric(f'loss_edge_type_{lang}', loss_edge_type, weight)
        loss_edge_norm = Metric(f'loss_edge_norm_{lang}', loss_edge_norm, weight)
        return loss_edge_type, loss_edge_norm
