import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

from arglib import add_argument, init_g_attr

from .transformer import TransformerModel, MultiHeadAttention


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
        # word2bpe (l_word x l_bpe) @ attn_bpe_output (l_bpe x bs x d) -> attn_word_output (l_word x bs x d)
        bs, l_word, l_bpe = word2bpe.shape
        attn_word_output = word2bpe @ attn_bpe_output
        return attn_word_output


@init_g_attr
class ContinuousRGCN(RGCNConv):
    """Similar to RGCN but with continous `edge_type` and `edge_norm`."""

    add_argument('num_bases', default=5, dtype=int, msg='number of bases for RGCN')

    def __init__(self, emb_dim, num_bases, num_relations):
        n_hid = emb_dim * 4
        super().__init__(n_hid, n_hid, num_bases, num_relations)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out * edge_norm.view(-1, 1)


@init_g_attr(default='property')
class GraphPredictor(nn.Module):
    """Based on the word-level representations, predict the edge types and the edge strengths (or edge norms)."""

    add_argument('num_relations', default=5, dtype=int, msg='number of distinct edge types.')

    def __init__(self, emb_dim, dropout, num_relations):
        super().__init__()
        self.norm_attn = MultiHeadAttention(8, emb_dim, dropout=dropout)
        self.type_attn = MultiHeadAttention(8, emb_dim, dropout=dropout)
        self.type_proj = nn.Linear(emb_dim, num_relations)

    def forward(self, h_in, word_mask):
        # Get norms first.
        norm_output, norm_attn_weights = self.norm_attn(h_in, word_mask, return_weights=True)
        # NOTE(j_luo) Sum all heads. # IDEA(j_luo) Summing might not be the best idea.
        norm_attn_weights = norm_attn_weights.sum(dim=1)
        norms = norm_attn_weights + norm_attn_weights.transpose(1, 2)  # Symmetrize attention weights.

        # Get types now.
        type_output, type_attn_weights = self.type_attn(h_in, word_mask, return_weights=True)
        # I'm decomposing the prediction.
        type_logits = self.type_proj(type_output)  # bs x l x nr
        type_logits = type_logits.unsqueeze(dim=2) + type_logits.unsqueeze(dim=1)
        type_probs = torch.log_softmax(type_logits, dim=-1).exp()

        return norms, type_probs


class Graphormer(TransformerModel):

    def __init__(self, params, dico, is_encoder, with_output):
        super().__init__(params, dico, is_encoder, with_output)
        self.assembler = Assembler()
        self.graph_predictor = GraphPredictor()
        self.rgcn = ContinuousRGCN()

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None, graph_info=None):
        assert graph_info is not None
        h = super().fwd(x, lengths, causal, src_enc=src_enc, src_len=src_len, positions=positions, langs=langs, cache=cache)
        h = h.transpose(0, 1)
        assembled_h = self.assembler(h, graph_info.bpe_mask, graph_info.word2bpe)
        norms, type_probs = self.graph_predictor(assembled_h, graph_info.word_mask)
        breakpoint()  # DEBUG(j_luo)
        # Prepare edge_norms and edge_types.
        # FIXME(j_luo)
        graph_h = self.rgcn(x=assembled_h, edge_norms=edge_norms, edge_types=edge_types)
        return graph_h
