from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from cs336_basics.transformer import ScaledDotProductAttentionLayer
from jaxtyping import Float
from torch import Tensor


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_block = tl.zeros((Q_TILE_SIZE, D), Q_block.dtype)
    l = tl.zeros((Q_TILE_SIZE,), Q_block.dtype)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), Q_block.dtype)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        temp_O_block = tl.zeros((Q_TILE_SIZE, D), Q_block.dtype)
        temp_S_block = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), Q_block.dtype)
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        temp_S_block = tl.dot(Q_block, tl.trans(K_block, (1, 0)), acc=temp_S_block)
        S_block = temp_S_block * scale
        if is_causal:
            q_range = (query_tile_index * Q_TILE_SIZE) + tl.arange(0, Q_TILE_SIZE)
            k_range = (key_tile_index * K_TILE_SIZE) + tl.arange(0, K_TILE_SIZE)
            mask = q_range[:, None] < k_range[None, :]
            mask_additive_value = tl.where(mask, -1e6, 0)
            S_block += mask_additive_value
        new_m = tl.maximum(m, tl.max(S_block, axis=-1))
        m_diff_exp = tl.exp(m - new_m)
        m = new_m
        P_block = tl.exp(S_block - new_m[:, None])
        l = m_diff_exp * l + tl.sum(P_block, axis=-1)
        temp_O_block = tl.dot(P_block.to(V_block.dtype), V_block, acc=temp_O_block)
        O_block = O_block * m_diff_exp[:, None] + temp_O_block
        # Advance blocks
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block = O_block * (1 / l)[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, O_block.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "batch_size n_queries d"],
        K: Float[Tensor, "batch_size n_keys d"],
        V: Float[Tensor, "batch_size n_keys d"],
        is_causal=False,
    ) -> Float[Tensor, "batch_size n_queries d"]:
        for ten in [Q, K, V]:
            assert ten.is_cuda, "Expected cuda tensors"
            assert ten.is_contiguous(), "must be contiguous"

        ctx.Q_TILE_SIZE = tl.constexpr(16)
        ctx.K_TILE_SIZE = tl.constexpr(16)
        ctx.BATCH_SIZE = Q.shape[0]
        ctx.NUM_QUERIES = Q.shape[1]
        ctx.NUM_KEYS = K.shape[1]
        ctx.DIM = K.shape[2]
        ctx.is_causal = is_causal
        O = torch.ones((Q.shape), device=Q.device)
        L = torch.empty((Q.shape[:-1]), device=Q.device)
        flash_fwd_kernel[(triton.cdiv(ctx.NUM_QUERIES, ctx.Q_TILE_SIZE), ctx.BATCH_SIZE)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            ctx.NUM_QUERIES,
            ctx.NUM_KEYS,
            1 / np.sqrt(ctx.DIM),
            tl.constexpr(ctx.DIM),
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            tl.constexpr(is_causal),
        )
        ctx.save_for_backward(Q, K, V, L)
        return O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


if __name__ == "__main__":
    attention = ScaledDotProductAttentionLayer()
    attn_impl = FlashAttentionTriton().apply
    device = "cuda"
    b = 1
    n_k = 16
    d = 16
    x = torch.tensor(15 * [0] + [1], device=device, dtype=torch.float32)
    rand = lambda: torch.randint(0, 2, (d, d), device=device, dtype=torch.float32)
    eye = lambda: torch.eye(d, device=device, dtype=torch.float32)
    diag = lambda: torch.diag(x)

    Q = diag()[None, :, :].expand(b, -1, -1).contiguous()
    K = eye()[None, :, :].expand(b, -1, -1).contiguous()
    V = eye()[None, :, :].expand(b, -1, -1).contiguous()
    O = attn_impl(Q, K, V)
    print(O)
    print(O.shape)
    rO = attention(Q, K, V)
    print(rO)
    print(rO.shape)
