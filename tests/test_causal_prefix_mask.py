"""
Test adapted from https://github.com/openai/triton/blob/0d7e7532279e45672555e344646f5c19c3972331/python/tutorials/06-fused-attention.py
"""
from contextlib import nullcontext
import math
import time

from scipy import stats

import torch

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def create_causal_mask(q: int, k: int, dtype: torch.dtype, device: torch.device):
    return (
        (torch.ones((q, k), device=device) - torch.inf).triu(k - q + 1).type(dtype)
    )


def attention_ref(q, k, v, sm_scale, causal, device):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    # for z in range(Z):
    #     for h in range(H):
    #         p[:, :, M == 0] = float("-inf")
    if causal:
        M = create_causal_mask(q.size(2), k.size(2), dtype=dtype, device=device)
        p += M
    p = torch.softmax(p.float(), dim=-1).type(dtype)
    ref_out = torch.matmul(p, v)
    return ref_out


torch.manual_seed(0)
repeats = 1
batch_size = 1
nheads = 1
seqlen = 16
n = 16
d = n // nheads
dropout_p = 0.0
causal = True
dtype = torch.bfloat16
device = 'cuda'
test_backward = True


with torch.inference_mode() if not test_backward else nullcontext():
    B = 8
    H = 12
    Q_N_CTX = 350 # 128 * 2 * 2
    KV_N_CTX = 350 * 100 # 256 * 2 * 2 * 2
    D_HEAD = 64

    torch.manual_seed(20)
    q = torch.empty((B, H, Q_N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0, std=.5)
    k = torch.empty((B, H, KV_N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0, std=.5)
    v = torch.empty((B, H, KV_N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0, std=.5)
    if test_backward:
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()
    cu_seqlens_q = torch.arange(
        0, (B + 1) * Q_N_CTX, step=Q_N_CTX, dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.arange(
        0, (B + 1) * KV_N_CTX, step=KV_N_CTX, dtype=torch.int32, device=device
    )

    s = time.time()
    flash_out = flash_attn_unpadded_func(
        q.transpose(1, 2).reshape(B * Q_N_CTX, H, D_HEAD),
        k.transpose(1, 2).reshape(B * KV_N_CTX, H, D_HEAD),
        v.transpose(1, 2).reshape(B * KV_N_CTX, H, D_HEAD),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=Q_N_CTX,
        max_seqlen_k=KV_N_CTX,
        dropout_p=dropout_p,
        causal=causal,
    )
    torch.cuda.synchronize()
    flash_took = time.time() - s
    s = time.time()
    ref_out = attention_ref(
        q, k, v, sm_scale=1/math.sqrt(D_HEAD), causal=causal, device=device
    ).transpose(1,2).reshape(B*Q_N_CTX, H, D_HEAD)
    torch.cuda.synchronize()
    ref_took = time.time() - s

    print("allclose", torch.allclose(flash_out, ref_out))
    print("max delta", (flash_out - ref_out).abs().max().item())
    print("relative max delta", ((flash_out - ref_out).abs().max() / ref_out.abs().mean()).item())
    print(stats.spearmanr(flash_out[0,0].float().detach().cpu().numpy(), ref_out[0,0].float().detach().cpu().numpy()))
    print(f"ref took: {ref_took:.5f}")
    print(f"flash attn took: {flash_took:.5f}")

    if test_backward:
        dout = torch.randn_like(q).transpose(1, 2).reshape(B * Q_N_CTX, H, D_HEAD)
        s = time.time()
        ref_out.backward(dout)
        torch.cuda.synchronize()
        ref_took = time.time() - s
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

        s = time.time()
        flash_out.backward(dout)
        torch.cuda.synchronize()
        flash_took = time.time() - s
        flash_dv, v.grad = v.grad.clone(), None
        flash_dk, k.grad = k.grad.clone(), None
        flash_dq, q.grad = q.grad.clone(), None

        for name, ref, flash in zip(
            ["dv", "dk", "dq"],
            [ref_dv, ref_dk, ref_dq],
            [flash_dv, flash_dk, flash_dq],
        ):
            print(f"=== evaling {name} ===")
            print("allclose", torch.allclose(flash, ref))
            print("max delta", (flash - ref).abs().max().item())
            print("relative max delta", ((flash - ref).abs().max() / ref.abs().mean()).item())
            print(stats.spearmanr(flash[0,0].flatten().float().detach().cpu().numpy(), ref[0,0].flatten().float().detach().cpu().numpy()))
        print(f"ref took: {ref_took:.5f}")
        print(f"flash attn took: {flash_took:.5f}")
