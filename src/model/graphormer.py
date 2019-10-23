from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

from arglib import add_argument, init_g_attr
from devlib import get_range

from .transformer import MultiHeadAttention, TransformerModel


@init_g_attr
class Assembler(MultiHeadAttention):
    """
    An Assembler instance composes one composite word embedding from its constituent BPE segments.
    """

    def __init__(self, emb_dim, dropout):
        super().__init__(8, emb_dim, dropout=dropout)

    def forward(self, h_in, bpe_mask, word2bpe):
        # Derive additive `attn_mask` from `bpe_mask`.
        attn_bpe_output, attn_bpe_output_weights = super().forward(h_in, bpe_mask, return_weights=True)
        # word2bpe (bs x l_word x l_bpe) @ attn_bpe_output (bs x l_bpe x d) -> attn_word_output (bs x l_word x d)
        bs, l_word, l_bpe = word2bpe.shape
        attn_word_output = word2bpe @ attn_bpe_output
        return attn_word_output


@init_g_attr
class ContinuousRGCN(RGCNConv):
    """Similar to RGCN but with continous `edge_type` and `edge_norm`."""

    add_argument('num_bases', default=5, dtype=int, msg='number of bases for RGCN')

    def __init__(self, emb_dim, num_bases, num_relations):
        super().__init__(emb_dim, emb_dim, num_bases, num_relations)
        self.basis = nn.Parameter(self.basis.transpose(0, 1).contiguous())  # NOTE(j_luo) This would help in message.

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        """The original algorithm consumes way too much memory. But we can use simple arithmetics to help.

        Let each basic be B(j), where j in {1..M}, and each relation r has its relation weight a(r, j), where r in {1...R}.
        Each edge e has its own relation distribution p(e, r) for e in E.

        Now for each relation r, the projection weight W(r) = sum_{j=1-J} a(r, j) * B(j).
        For each edge e, the projection weight is W(e) = sum_{r=1-R} p(e, r) * W(r).
        The output now is :
                    h'(e) = h(e) @ W(e)
                          = h(e) @ (sum_{r=1-R} p(e, r) * W(r))
                          = h(e) @ (sum_{r=1-R, j=1-J} p(e, r) * a(r, j) * B(j))
                          = sum_{r=1-R, j=1-J} [p(e, r) * a(r, j)] * [h(e) @ B(j)]
                          = sum_{j=1-J} c(e, j) * [h(e) @ B(j)],
        where:
                  c(e, j) = sum_{r=1-R} p(e, r) * a(r, j)

        """
        E, _ = x_j.shape
        h_e_basis = x_j @ self.basis.view(self.in_channels, self.num_bases * self.out_channels)
        h_e_basis = h_e_basis.view(E, self.num_bases, self.out_channels)

        weight = edge_type @ self.att  # size: E x nr @ nr x nb -> E x nb
        # size: E x 1 x nb @ E x nb x n_out -> E x 1 x n_out -> E x n_out
        out = (weight.unsqueeze(dim=1) @ h_e_basis).squeeze(dim=-2)
        return out * edge_norm.view(-1, 1)


@init_g_attr(default='property')
class GraphPredictor(nn.Module):
    """Based on the word-level representations, predict the edge types and the edge strengths (or edge norms)."""

    add_argument('num_relations', default=5, dtype=int, msg='number of distinct edge types.')
    add_argument('edge_norm_agg', default='sum', choices=[
                 'sum', 'mean'], dtype=str, msg='how to aggregate the attention scores to get an edge norm.')

    def __init__(self, emb_dim, dropout, num_relations, edge_norm_agg):
        super().__init__()
        self.norm_attn = MultiHeadAttention(8, emb_dim, dropout=dropout)
        self.type_attn = MultiHeadAttention(8, emb_dim, dropout=dropout)
        self.type_proj = nn.Linear(emb_dim, num_relations)

    def forward(self, h_in, word_mask):
        # Get norms first.
        norm_output, norm_attn_weights = self.norm_attn(h_in, word_mask, return_weights=True)
        # NOTE(j_luo) Aggregate attention scores to get an edge norm.
        if self.edge_norm_agg == 'sum':
            norm_attn_weights = norm_attn_weights.sum(dim=1)
        else:
            norm_attn_weights = norm_attn_weights.mean(dim=1)

        norms = norm_attn_weights + norm_attn_weights.transpose(1, 2)  # Symmetrize attention weights.

        # Get types now.
        type_output, type_attn_weights = self.type_attn(h_in, word_mask, return_weights=True)
        # I'm decomposing the prediction.
        type_logits = self.type_proj(type_output)  # bs x wl x nr
        type_logits = type_logits.unsqueeze(dim=2) + type_logits.unsqueeze(dim=1)
        type_probs = torch.log_softmax(type_logits, dim=-1).exp()

        return norms, type_probs


Tensor = torch.Tensor


@dataclass
class GraphData:
    node_features: Tensor
    edge_index: Tensor
    edge_norm: Tensor
    edge_type: Tensor


class Graphormer(TransformerModel):

    add_argument('ablation_mode', default='full', dtype=str, choices=['full', 'ffn', 'none', 'self_attn'],
                 msg='ablation mode. full means full model, ffn replaces rgcn with ffn, and none means using assembler only.')
    add_argument("self_attn_layers", default=0, dtype=int, msg='number of layers for self attention layers.')

    def __init__(self, params, dico, is_encoder, with_output):
        super().__init__(params, dico, is_encoder, with_output)
        self.assembler = Assembler()
        self.ablation_mode = params.ablation_mode

        if self.ablation_mode == 'full':
            self.graph_predictor = GraphPredictor()
            self.rgcn = ContinuousRGCN()
        elif self.ablation_mode == 'ffn':
            # NOTE(j_luo) This actually is about 10x smaller than 'full' mode.
            self.linear = nn.Linear(params.emb_dim, params.emb_dim)
        elif self.ablation_mode == 'none':
            pass
        else:
            if not params.self_attn_layers > 0:
                raise ValueError(f'Must have at least one layer.')
            layers = [MultiHeadAttention(8, params.emb_dim, dropout=params.dropout)
                      for _ in range(params.self_attn_layers)]
            self.self_attn_layers = nn.Sequential(*layers)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None, graph_info=None, return_graph_data=False):
        assert graph_info is not None
        assert not return_graph_data or self.ablation_mode == 'full'

        h = super().fwd(x, lengths, causal, src_enc=src_enc, src_len=src_len, positions=positions, langs=langs, cache=cache)
        h = h.transpose(0, 1)
        assembled_h = self.assembler(h, graph_info.bpe_mask, graph_info.word2bpe)
        if self.ablation_mode == 'full':
            norms, type_probs = self.graph_predictor(assembled_h, graph_info.word_mask)
            # Prepare node_features, edge_index, edge_norm and edge_type.
            graph_data = self._prepare_for_geometry(assembled_h, norms, type_probs)
            output = self.rgcn(x=graph_data.node_features, edge_index=graph_data.edge_index,
                               edge_norm=graph_data.edge_norm, edge_type=graph_data.edge_type)
            # Now reshape output for later usage. Note that the length dimension has changed to represent words instead of BPEs.
            bs, wl, _ = assembled_h.shape
            output = output.view(bs, wl, -1)
        elif self.ablation_mode == 'ffn':
            output = self.linear(assembled_h)
        elif self.ablation_mode == 'none':
            output = assembled_h
        else:
            output = assembled_h
            for layer in self.self_attn_layers:
                output = layer(output, graph_info.word_mask)

        output = output.transpose(0, 1)
        if return_graph_data:
            return output, graph_data
        else:
            return output

    def _prepare_for_geometry(self, assembled_h, norms, type_probs):
        """
        inputs:
            assembled_h:    bs x wl x d
            norms:          bs x wl x wl
            type_probs:     bs x wl x wl x nr

        outputs:
            node_features:  V x d
            edge_index:     2 x E
            edge_norms:     E
            edge_types:     E x nr
        where V = bs * wl and E = bs * wl * wl.
        """
        bs, wl, _, nr = type_probs.shape
        V = bs * wl
        E = bs * wl * wl

        # node_features is just a reshaped version of assembled_h.
        node_features = assembled_h.view(V, -1)

        # edge_index is a collection of fully connected graphs, each of which corresponds to one sentence.
        edge_index_offset = get_range(bs, 1, 0) * wl  # bs
        edge_index_i = get_range(wl, 2, 0).expand(wl, wl)  # wl x 1 -> wl x wl
        edge_index_i = edge_index_offset.view(bs, 1, 1) + edge_index_i
        edge_index_j = get_range(wl, 2, 1).expand(wl, wl)  # 1 x wl -> wl x wl
        edge_index_j = edge_index_offset.view(bs, 1, 1) + edge_index_j
        edge_index = torch.stack([edge_index_i, edge_index_j], dim=0).view(2, E)

        # edge_norms is just a reshaped version of norms.
        edge_norms = norms.view(E)

        # edge_types is similar.
        edge_types = type_probs.view(E, nr)

        return GraphData(node_features, edge_index, edge_norms, edge_types)
