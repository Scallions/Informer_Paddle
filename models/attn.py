import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np

from math import sqrt
from ..utils.masking import TriangularCausalMask, ProbMask
from ..utils.tools import swap_shape


class FullAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = paddle.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=paddle.device.get_device())

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            scores = paddle.where(attn_mask.mask.tile([1,H,1,1]), -np.inf*paddle.ones_like(scores), scores)

        A = self.dropout(F.softmax(scale * scores, axis=-1))
        V = paddle.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V, A)
        else:
            return (V, None)

class ProbAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand([B, H, L_Q, L_K, E])
        index_sample = paddle.randint(high=L_K, shape=[L_Q, sample_k]) # real U = U_part(factor*ln(L_k))*L_q
        index_sample = index_sample.reshape([1,1,L_Q, sample_k, 1]).tile([B, H, 1, 1, E])
        # K_sample = K_expand[:, :, paddle.arange(L_Q).unsqueeze(1), index_sample, :]
        K_sample = K_expand.index_sample(index_sample)
        Q_K_sample = paddle.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - paddle.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[paddle.arange(B)[:, None, None],
                     paddle.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = paddle.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(axis=-2)
            contex = V_sum.unsqueeze(-2).expand([B, H, L_Q, V_sum.shape[-1]]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=paddle.device.get_device())
            # scores.masked_fill_(attn_mask.mask, -np.inf)
            scores = paddle.where(attn_mask.mask, -np.inf*paddle.ones_like(scores), scores)
        attn = F.softmax(scores, axis=-1) # nn.Softmax(dim=-1)(scores)

        context_in[paddle.arange(B)[:, None, None],
                   paddle.arange(H)[None, :, None],
                   index, :] = paddle.matmul(attn, V).cast(context_in.dtype)
        if self.output_attention:
            attns = (paddle.ones([B, H, L_V, L_V])/L_V).cast(attn.dtype)#.to(attn.device)
            attns[paddle.arange(B)[:, None, None], paddle.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(swap_shape(queries, 2,1))
        keys = keys.transpose(swap_shape(keys, 2,1))
        values = values.transpose(swap_shape(values, 2,1))

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(swap_shape(context, 2,1)), attn


class AttentionLayer(nn.Layer):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(swap_shape(out, 2,1))#.contiguous()
        out = out.reshape([B, L, -1])

        return self.out_projection(out), attn
