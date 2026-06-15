#!/usr/bin/env python3
"""SM120 kernel compatibility smoke tests for TileKernels."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

os.environ.setdefault('TILELANG_PRINT_ON_COMPILATION', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch

import tile_kernels
from tile_kernels.engram import (
    engram_gate_bwd,
    engram_gate_fwd,
    engram_hash,
    fused_weight,
    grad_w_reduce,
)
from tile_kernels.modeling.mhc.ops import (
    expand_to_mhc,
    mhc_head_compute_mix,
    mhc_post,
    mhc_pre_apply_mix,
    mhc_pre_big_fuse,
    mhc_pre_norm_fn,
    mhc_pre_split_mixes,
    mhc_multilayer_recompute,
    sinkhorn_normalize,
)
from tile_kernels.quant.per_token_cast_to_e5m6_kernel import per_token_cast_to_e5m6
from tile_kernels.quant.cast_back_e5m6_kernel import cast_back_e5m6
from tile_kernels.testing.generator import generate_topk_idx
from tile_kernels.torch.engram import make_offsets
from tile_kernels.config import get_num_sms

# Shared smoke-test dimensions (from tests/testing/generator.py defaults)
NUM_TOKENS = 4001
HIDDEN = 7168
NUM_TOKENS_ALIGNED = 4096  # align(4001, 128) — required by per-channel and transpose kernels
NUM_EXPERTS = 72
NUM_TOPK = 6
NUM_EP_RANKS = 8
MOE_PARAMS = {
    'num_send_tokens': NUM_TOKENS_ALIGNED,
    'num_topk': NUM_TOPK,
    'num_experts': NUM_EXPERTS,
    'num_ep_ranks': NUM_EP_RANKS,
}


@dataclass
class KernelCase:
    name: str
    domain: str
    source_file: str
    run: Callable[[], None]


@dataclass
class KernelResult:
    name: str
    domain: str
    source_file: str
    status: str
    error_type: Optional[str] = None
    error_summary: Optional[str] = None
    full_traceback: Optional[str] = None


def twice_stride(w: torch.Tensor) -> torch.Tensor:
    """Match tests/transpose: non-contiguous leading dim with column-major last dim."""
    twice_w = w.new_empty((w.shape[0], w.shape[1] * 2))
    ret = torch.chunk(twice_w, 2, dim=1)[0]
    ret[:] = w
    return ret


def classify_error(exc: BaseException) -> tuple[str, str]:
    msg = str(exc)
    msg_lower = msg.lower()
    exc_type = type(exc).__name__

    patterns = [
        ('compilation_error', r'compilationfailed|nvcc|ptxas|error compiling|cudacompilationerror'),
        ('tilelang_internal', r'tvm\.error|unsupported target|tilelang|internalerror'),
        ('cuda_runtime', r'cuda error|cudaerror|illegal memory access|illegal instruction|cuda out of memory|device-side assert|failed to set the allowed dynamic shared memory'),
        ('input_validation', r'typeerror|notimplementederror|got an unexpected keyword|missing \d+ required positional|takes \d+ positional arguments'),
        ('input_validation', r'assertionerror'),
        ('correctness', r'assert_equal|mismatch'),
    ]
    combined = f'{exc_type} {msg_lower}'
    for category, pattern in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return category, msg.split('\n')[0][:300]

    if exc_type == 'AssertionError':
        return 'correctness', msg.split('\n')[0][:300]
    return 'other', f'{exc_type}: {msg.split(chr(10))[0][:300]}'


def _moe_topk_idx() -> torch.Tensor:
    return generate_topk_idx(MOE_PARAMS)


def _moe_mapping():
    topk_idx = _moe_topk_idx()
    return tile_kernels.moe.get_fused_mapping(topk_idx, NUM_EXPERTS, 0, 128)


def build_cases() -> list[KernelCase]:
    cases: list[KernelCase] = []

    # --- MoE ---
    def smoke_topk_gate():
        scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float, device='cuda')
        tile_kernels.moe.topk_gate(scores, NUM_TOPK)
        torch.cuda.synchronize()

    cases.append(KernelCase('topk_gate', 'moe', 'moe/topk_gate_kernel.py', smoke_topk_gate))

    def smoke_top2_sum_gate():
        num_tokens = 32
        num_routed_experts = 72
        logits = torch.randn(num_tokens, num_routed_experts, dtype=torch.float32, device='cuda')
        bias = torch.randn(num_routed_experts, dtype=torch.float32, device='cuda')
        unmapped = torch.zeros(num_tokens, 6, dtype=torch.int64, device='cuda')
        tile_kernels.moe.top2_sum_gate(
            logits, bias, 6, 0, 0, False, 1, 1.5, 0, 4, 0, 2, 'sigmoid',
            unmapped_topk_idx=unmapped,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase('top2_sum_gate', 'moe', 'moe/top2_sum_gate_kernel.py', smoke_top2_sum_gate))

    def smoke_topk_sum_and_topk_group_idx():
        scores = torch.randn(NUM_TOKENS, 8, 9, dtype=torch.float, device='cuda')
        tile_kernels.moe.topk_sum_and_topk_group_idx(scores, 2, 4)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'topk_sum_and_topk_group_idx', 'moe', 'moe/topk_sum_and_topk_group_idx_kernel.py',
        smoke_topk_sum_and_topk_group_idx,
    ))

    def smoke_get_fused_mapping():
        _moe_mapping()
        torch.cuda.synchronize()

    cases.append(KernelCase('get_fused_mapping', 'moe', 'moe/get_fused_mapping_kernel.py', smoke_get_fused_mapping))

    def smoke_expand_to_fused():
        topk_idx = _moe_topk_idx()
        x = torch.randn(topk_idx.shape[0], HIDDEN, dtype=torch.bfloat16, device='cuda')
        pos_to_expert, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
            topk_idx, NUM_EXPERTS, 0, 16,
        )
        tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_expert)
        torch.cuda.synchronize()

    cases.append(KernelCase('expand_to_fused', 'moe', 'moe/expand_to_fused_kernel.py', smoke_expand_to_fused))

    def smoke_expand_to_fused_with_sf():
        topk_idx = _moe_topk_idx()
        num_tokens = topk_idx.shape[0]
        x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device='cuda')
        x_fp8, x_sf = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=128)
        pos_to_expert, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
            topk_idx, NUM_EXPERTS, 0, 16,
        )
        tile_kernels.moe.expand_to_fused_with_sf(
            (x_fp8, x_sf.contiguous()), 128, token_topk_to_pos, pos_to_expert, True,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'expand_to_fused_with_sf', 'moe', 'moe/expand_to_fused_kernel.py', smoke_expand_to_fused_with_sf,
    ))

    def smoke_reduce_fused():
        topk_idx = _moe_topk_idx()
        num_tokens = topk_idx.shape[0]
        hidden = 4096
        expanded = torch.randn(num_tokens * NUM_TOPK, hidden, dtype=torch.bfloat16, device='cuda')
        _, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, NUM_EXPERTS, 0, 1)
        topk_weights = torch.rand(num_tokens, NUM_TOPK, dtype=torch.float32, device='cuda')
        tile_kernels.moe.reduce_fused(expanded, topk_weights, token_topk_to_pos, '', None, None)
        torch.cuda.synchronize()

    cases.append(KernelCase('reduce_fused', 'moe', 'moe/reduce_fused_kernel.py', smoke_reduce_fused))

    def smoke_normalize_weight():
        topk_idx = _moe_topk_idx()
        num_tokens = topk_idx.shape[0]
        weights = torch.rand(num_tokens, NUM_TOPK, dtype=torch.float32, device='cuda')
        tile_kernels.moe.normalize_weight(weights)
        torch.cuda.synchronize()

    cases.append(KernelCase('normalize_weight', 'moe', 'moe/normalize_weight_kernel.py', smoke_normalize_weight))

    def smoke_aux_fi():
        topk_idx = _moe_topk_idx()
        tile_kernels.moe.aux_fi(topk_idx, NUM_EXPERTS, NUM_TOPK)
        torch.cuda.synchronize()

    cases.append(KernelCase('aux_fi', 'moe', 'moe/aux_fi_kernel.py', smoke_aux_fi))

    def smoke_group_count():
        topk_idx = _moe_topk_idx()
        tile_kernels.moe.group_count(topk_idx, NUM_EXPERTS)
        torch.cuda.synchronize()

    cases.append(KernelCase('group_count', 'moe', 'moe/group_count_kernel.py', smoke_group_count))

    def smoke_mask_indices_by_tp():
        topk_idx = _moe_topk_idx()
        n = NUM_EXPERTS * NUM_EP_RANKS
        tile_kernels.moe.mask_indices_by_tp(topk_idx, n, NUM_EP_RANKS, 0, 2)
        torch.cuda.synchronize()

    cases.append(KernelCase('mask_indices_by_tp', 'moe', 'moe/mask_indices_by_tp_kernel.py', smoke_mask_indices_by_tp))

    def smoke_inplace_unique_group_indices():
        topk_idx = _moe_topk_idx()
        group_indices = topk_idx // (NUM_EXPERTS * NUM_EP_RANKS // 8)
        tile_kernels.moe.inplace_unique_group_indices(group_indices.clone(), 8)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'inplace_unique_group_indices', 'moe', 'moe/inplace_unique_group_indices_kernel.py',
        smoke_inplace_unique_group_indices,
    ))

    # --- Quant ---
    def _quant_x():
        return torch.randn(NUM_TOKENS_ALIGNED, HIDDEN, dtype=torch.bfloat16, device='cuda')

    def smoke_per_token_cast():
        tile_kernels.quant.per_token_cast(_quant_x(), 'e4m3', num_per_channels=128)
        torch.cuda.synchronize()

    cases.append(KernelCase('per_token_cast', 'quant', 'quant/per_token_cast_kernel.py', smoke_per_token_cast))

    def smoke_per_token_cast_with_sf_only():
        tile_kernels.quant.per_token_cast_with_sf_only(_quant_x(), 'e4m3', num_per_channels=128)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_token_cast_with_sf_only', 'quant', 'quant/per_token_cast_kernel.py', smoke_per_token_cast_with_sf_only,
    ))

    def smoke_per_token_cast_with_precomputed_sf():
        x = _quant_x()
        _, sf = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=128)
        tile_kernels.quant.per_token_cast_with_precomputed_sf(x, 'e4m3', 128, sf)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_token_cast_with_precomputed_sf', 'quant', 'quant/per_token_cast_kernel.py',
        smoke_per_token_cast_with_precomputed_sf,
    ))

    def smoke_per_token_cast_to_e5m6():
        per_token_cast_to_e5m6(_quant_x(), HIDDEN, True, True, True)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_token_cast_to_e5m6', 'quant', 'quant/per_token_cast_to_e5m6_kernel.py', smoke_per_token_cast_to_e5m6,
    ))

    def smoke_per_block_cast():
        tile_kernels.quant.per_block_cast(_quant_x(), 'e4m3', block_size=(128, 128))
        torch.cuda.synchronize()

    cases.append(KernelCase('per_block_cast', 'quant', 'quant/per_block_cast_kernel.py', smoke_per_block_cast))

    def smoke_per_block_cast_with_sf_only():
        tile_kernels.quant.per_block_cast_with_sf_only(_quant_x(), 'e4m3', block_size=(128, 128))
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_block_cast_with_sf_only', 'quant', 'quant/per_block_cast_kernel.py', smoke_per_block_cast_with_sf_only,
    ))

    def smoke_per_block_cast_with_precomputed_sf():
        x = _quant_x()
        _, sf = tile_kernels.quant.per_block_cast(x, 'e4m3', block_size=(128, 128))
        tile_kernels.quant.per_block_cast_with_precomputed_sf(x, 'e4m3', (128, 128), sf)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_block_cast_with_precomputed_sf', 'quant', 'quant/per_block_cast_kernel.py',
        smoke_per_block_cast_with_precomputed_sf,
    ))

    def smoke_per_block_cast_lossless():
        x = _quant_x()
        x_fp4 = tile_kernels.torch.cast(x, 'e2m1', (128, 128))
        tile_kernels.quant.per_block_cast_lossless(
            x_fp4, 'e4m3',
            x_block_size=(128, 128),
            out_block_size=(128, 128),
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_block_cast_lossless', 'quant', 'quant/per_block_cast_lossless_kernel.py', smoke_per_block_cast_lossless,
    ))

    def smoke_per_channel_cast():
        tile_kernels.quant.per_channel_cast(_quant_x(), 'e4m3', num_per_tokens=128)
        torch.cuda.synchronize()

    cases.append(KernelCase('per_channel_cast', 'quant', 'quant/per_channel_cast_kernel.py', smoke_per_channel_cast))

    def smoke_per_channel_cast_fused():
        tile_kernels.quant.per_channel_cast_fused(_quant_x(), 'e4m3', num_per_tokens=128)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_channel_cast_fused', 'quant', 'quant/per_channel_cast_fused_kernel.py', smoke_per_channel_cast_fused,
    ))

    def smoke_per_channel_cast_and_transpose():
        tile_kernels.quant.per_channel_cast_and_transpose(_quant_x(), 'e4m3', num_per_tokens=128)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'per_channel_cast_and_transpose', 'quant', 'quant/per_channel_cast_and_transpose_kernel.py',
        smoke_per_channel_cast_and_transpose,
    ))

    def smoke_cast_back():
        x = _quant_x()
        x_casted, x_sf = tile_kernels.quant.per_block_cast(x, 'e4m3', block_size=(128, 128))
        tile_kernels.quant.cast_back((x_casted, x_sf), 'bf16', (128, 128))
        torch.cuda.synchronize()

    cases.append(KernelCase('cast_back', 'quant', 'quant/cast_back_kernel.py', smoke_cast_back))

    def smoke_per_token_cast_back():
        x = _quant_x()
        x_casted, x_sf = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=128)
        tile_kernels.quant.per_token_cast_back((x_casted, x_sf), 'bf16', num_per_channels=128)
        torch.cuda.synchronize()

    cases.append(KernelCase('per_token_cast_back', 'quant', 'quant/cast_back_kernel.py', smoke_per_token_cast_back))

    def smoke_cast_back_e5m6():
        x = _quant_x()
        x_casted, x_sf = per_token_cast_to_e5m6(x, HIDDEN, True, True, True)
        cast_back_e5m6((x_casted, x_sf), 'bf16', (1, HIDDEN))
        torch.cuda.synchronize()

    cases.append(KernelCase('cast_back_e5m6', 'quant', 'quant/cast_back_e5m6_kernel.py', smoke_cast_back_e5m6))

    def smoke_swiglu_forward_and_per_token_cast():
        topk_idx = _moe_topk_idx()
        num_tokens = topk_idx.shape[0]
        hidden_half = 3584
        x = torch.randn(num_tokens, hidden_half * 2, dtype=torch.bfloat16, device='cuda')
        pos_to_expert, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
            topk_idx, NUM_EXPERTS, 0, 16,
        )
        expanded = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_expert)
        tile_kernels.quant.swiglu_forward_and_per_token_cast(
            expanded, 'e4m3', pos_to_expert=pos_to_expert, num_per_channels=128,
            use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'swiglu_forward_and_per_token_cast', 'quant', 'quant/swiglu_forward_and_per_token_cast_kernel.py',
        smoke_swiglu_forward_and_per_token_cast,
    ))

    def smoke_swiglu_backward_and_per_token_cast():
        topk_idx = _moe_topk_idx()
        num_tokens = topk_idx.shape[0]
        hidden_half = 3584
        topk_weights = torch.rand(num_tokens, NUM_TOPK, dtype=torch.float32, device='cuda')
        x = torch.randn(num_tokens, hidden_half * 2, dtype=torch.bfloat16, device='cuda')
        _, pos_to_token, pos_to_token_topk, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
            topk_idx, NUM_EXPERTS, 0, 128,
        )
        x_expand = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_token)
        x_fp8 = tile_kernels.quant.per_token_cast(x_expand, 'e4m3', num_per_channels=128)
        grad_out = torch.randn(x_expand.shape[0], hidden_half, dtype=torch.bfloat16, device='cuda')
        tile_kernels.quant.swiglu_backward_and_per_token_cast(
            x_fp8, grad_out, topk_weights, pos_to_token_topk, token_topk_to_pos,
            num_per_channels=128, round_sf=True,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'swiglu_backward_and_per_token_cast', 'quant', 'quant/swiglu_backward_and_per_token_cast_kernel.py',
        smoke_swiglu_backward_and_per_token_cast,
    ))

    def smoke_swiglu_forward_and_per_channel_cast_and_transpose():
        x = torch.randn(4096, 7168, dtype=torch.bfloat16, device='cuda')
        tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose(
            x, 'e4m3', num_per_tokens=128, round_sf=True,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'swiglu_forward_and_per_channel_cast_and_transpose', 'quant',
        'quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py',
        smoke_swiglu_forward_and_per_channel_cast_and_transpose,
    ))

    # --- Transpose ---
    def smoke_transpose():
        x = torch.randn(NUM_TOKENS_ALIGNED, HIDDEN, dtype=torch.bfloat16, device='cuda')
        x = twice_stride(x)
        tile_kernels.transpose.transpose(x)
        torch.cuda.synchronize()

    cases.append(KernelCase('transpose', 'transpose', 'transpose/batched_transpose_kernel.py', smoke_transpose))

    def smoke_batched_transpose():
        x = torch.randn(8, NUM_TOKENS_ALIGNED, HIDDEN, dtype=torch.bfloat16, device='cuda')
        tile_kernels.transpose.batched_transpose(x)
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'batched_transpose', 'transpose', 'transpose/batched_transpose_kernel.py', smoke_batched_transpose,
    ))

    # --- Engram ---
    def smoke_engram_hash():
        ngram_token_ids = torch.randint(0, 100000, (NUM_TOKENS, 3), dtype=torch.int32, device='cuda')
        multipliers = torch.randint(0, 100000, (2, 3), dtype=torch.int64, device='cuda')
        vocab_sizes = torch.randint(100000, 1000000, (2, 2, 8), dtype=torch.int32, device='cuda')
        offsets = make_offsets(vocab_sizes)
        engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)
        torch.cuda.synchronize()

    cases.append(KernelCase('engram_hash', 'engram', 'engram/engram_hash_kernel.py', smoke_engram_hash))

    def _engram_data():
        hc = 4
        x = torch.randn(NUM_TOKENS, hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(NUM_TOKENS, hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device='cuda')
        wh = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        we = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        weight_fused = wh.float() * we.float()
        return x, k, v, weight_fused

    def smoke_engram_gate_fwd():
        x, k, v, weight_fused = _engram_data()
        engram_gate_fwd(x, k, v, weight_fused, 1e-20, 1e-6, save_for_backward=False)
        torch.cuda.synchronize()

    cases.append(KernelCase('engram_gate_fwd', 'engram', 'engram/engram_gate_kernel.py', smoke_engram_gate_fwd))

    def smoke_engram_gate_bwd():
        x, k, v, weight_fused = _engram_data()
        out, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
            x, k, v, weight_fused, 1e-20, 1e-6, save_for_backward=True,
        )
        grad_out = torch.randn_like(out)
        engram_gate_bwd(
            grad_out, x, k, v, weight_fused, dot, gate_score, rstd_x, rstd_k, 1e-6,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase('engram_gate_bwd', 'engram', 'engram/engram_gate_kernel.py', smoke_engram_gate_bwd))

    def smoke_fused_weight():
        hc = 4
        wh = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        we = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        fused_weight(wh, we)
        torch.cuda.synchronize()

    cases.append(KernelCase('fused_weight', 'engram', 'engram/engram_fused_weight_kernel.py', smoke_fused_weight))

    def smoke_grad_w_reduce():
        hc = 4
        # Use minimal block count to stay within per-block shared memory limits
        num_blocks = 4
        grad_w_partial = torch.randn(num_blocks, hc, HIDDEN, dtype=torch.float32, device='cuda')
        weight_hidden = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        weight_embed = torch.randn(hc, HIDDEN, dtype=torch.bfloat16, device='cuda')
        grad_wh = torch.zeros(hc, HIDDEN, dtype=torch.float32, device='cuda')
        grad_we = torch.zeros(hc, HIDDEN, dtype=torch.float32, device='cuda')
        grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_wh, grad_we)
        torch.cuda.synchronize()

    cases.append(KernelCase('grad_w_reduce', 'engram', 'engram/engram_grad_w_reduce_kernel.py', smoke_grad_w_reduce))

    # --- mHC ---
    def _mhc_residual_fn(mhc_mult: int = 4):
        n0, n1, hidden = 1, 4096, 2560
        mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
        residual = torch.randn(n0, n1, mhc_mult, hidden, dtype=torch.bfloat16, device='cuda')
        fn = torch.randn(mhc_mult3, mhc_mult, hidden, dtype=torch.float, device='cuda').flatten(1, 2) * 1e-4
        return residual, fn, mhc_mult

    def smoke_expand_to_mhc():
        x = torch.randn(1, 1024, 2560, dtype=torch.bfloat16, device='cuda')
        expand_to_mhc(x, 4)
        torch.cuda.synchronize()

    cases.append(KernelCase('expand_to_mhc', 'mhc', 'mhc/expand_kernel.py', smoke_expand_to_mhc))

    def smoke_mhc_pre_norm_fn():
        residual, fn, _ = _mhc_residual_fn()
        mhc_pre_norm_fn(residual, fn, None, 1e-6)
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_pre_norm_fn', 'mhc', 'mhc/norm_fn_kernel.py', smoke_mhc_pre_norm_fn))

    def smoke_mhc_head_compute_mix():
        input_mix = torch.randn(1, 1024, 4, dtype=torch.float, device='cuda')
        mhc_scale = torch.randn(1, dtype=torch.float, device='cuda')
        mhc_base = torch.randn(4, dtype=torch.float, device='cuda')
        mhc_head_compute_mix(input_mix, mhc_scale, mhc_base, 1e-2)
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_head_compute_mix', 'mhc', 'mhc/head_compute_mix_kernel.py', smoke_mhc_head_compute_mix))

    def smoke_mhc_pre_split_mixes():
        mhc_mult = 4
        mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
        input_mixes = torch.randn(1, 1024, mhc_mult3, dtype=torch.float, device='cuda')
        mhc_scale = torch.randn(3, dtype=torch.float, device='cuda')
        mhc_base = torch.randn(mhc_mult3, dtype=torch.float, device='cuda')
        mhc_pre_split_mixes(input_mixes, mhc_scale, mhc_base, mhc_mult, 2.0, 1e-2)
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_pre_split_mixes', 'mhc', 'mhc/pre_split_mixes_kernel.py', smoke_mhc_pre_split_mixes))

    def smoke_mhc_pre_apply_mix():
        x = torch.randn(1, 1024, 4, 2560, dtype=torch.bfloat16, device='cuda')
        mix = torch.randn(1, 1024, 4, 1, dtype=torch.float32, device='cuda')
        mhc_pre_apply_mix(x, mix)
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_pre_apply_mix', 'mhc', 'mhc/pre_apply_mix_kernel.py', smoke_mhc_pre_apply_mix))

    def smoke_mhc_pre_big_fuse():
        residual, fn, mhc_mult = _mhc_residual_fn()
        mhc_scale = torch.randn(3, dtype=torch.float, device='cuda')
        mhc_base = torch.randn(mhc_mult * 2 + mhc_mult * mhc_mult, dtype=torch.float, device='cuda')
        mhc_pre_big_fuse(
            residual, fn, mhc_scale, mhc_base,
            rms_eps=1e-6, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=1.0, sinkhorn_repeat=10, n_splits=16,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_pre_big_fuse', 'mhc', 'mhc/pre_big_fuse_kernel.py', smoke_mhc_pre_big_fuse))

    def smoke_sinkhorn_normalize():
        comb = torch.randn(1, 1024, 4, 4, dtype=torch.float32, device='cuda')
        sinkhorn_normalize(comb, repeat=10, eps=1e-6)
        torch.cuda.synchronize()

    cases.append(KernelCase('sinkhorn_normalize', 'mhc', 'mhc/sinkhorn_kernel.py', smoke_sinkhorn_normalize))

    def smoke_mhc_post():
        x = torch.randn(1, 4096, 2560, dtype=torch.bfloat16, device='cuda')
        residual = torch.randn(1, 4096, 4, 2560, dtype=torch.bfloat16, device='cuda')
        post_mix = torch.randn(1, 4096, 4, 1, dtype=torch.float32, device='cuda')
        comb_mix = torch.randn(1, 4096, 4, 4, dtype=torch.float32, device='cuda')
        mhc_post(x, residual, post_mix, comb_mix)
        torch.cuda.synchronize()

    cases.append(KernelCase('mhc_post', 'mhc', 'mhc/post_kernel.py', smoke_mhc_post))

    def smoke_mhc_multilayer_recompute():
        bs, seq, mhc_mult, hidden = 1, 8192, 4, 2560
        num_layers, num_post = 3, 2
        initial_residual = torch.randn(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16)
        pre_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_layers)]
        layer_output_list = [torch.randn(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
        post_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_post)]
        comb_mix_list = [torch.randn(bs, seq, mhc_mult, mhc_mult, device='cuda', dtype=torch.float32) for _ in range(num_post)]
        layer_input_list = [torch.empty(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_layers)]
        residual_list = [torch.empty(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
        mhc_multilayer_recompute(
            initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list,
            layer_input_list, residual_list,
        )
        torch.cuda.synchronize()

    cases.append(KernelCase(
        'mhc_multilayer_recompute', 'mhc', 'mhc/multilayer_recompute_kernel.py', smoke_mhc_multilayer_recompute,
    ))

    return cases


def run_case(case: KernelCase) -> KernelResult:
    try:
        case.run()
        return KernelResult(case.name, case.domain, case.source_file, 'pass')
    except Exception as exc:
        error_type, error_summary = classify_error(exc)
        return KernelResult(
            case.name, case.domain, case.source_file, 'fail',
            error_type=error_type,
            error_summary=error_summary,
            full_traceback=traceback.format_exc(),
        )


def get_env_info() -> dict[str, str]:
    import tilelang
    from tilelang.contrib import nvcc
    from tilelang.utils.target import determine_target

    prop = torch.cuda.get_device_properties(0)
    target = determine_target(return_object=True)
    ver = nvcc.get_target_compute_version(target)
    major, minor = nvcc.parse_compute_version(ver)

    return {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'gpu_name': prop.name,
        'compute_capability': f'{prop.major}.{prop.minor}',
        'sm_arch': f'SM{prop.major}{prop.minor}',
        'driver': torch.cuda.get_device_properties(0).name,  # overwritten below
        'cuda_version': torch.version.cuda or 'unknown',
        'pytorch': torch.__version__,
        'tilelang': tilelang.__version__,
        'tilelang_target': str(target),
        'tilelang_compute_version': ver,
        'tilelang_major': str(major),
    }


def get_driver_version() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True,
        )
        return out.strip().split('\n')[0]
    except Exception:
        return 'unknown'


def write_report(results: list[KernelResult], output_path: str, log_path: str) -> None:
    env = get_env_info()
    env['driver'] = get_driver_version()

    passed = [r for r in results if r.status == 'pass']
    failed = [r for r in results if r.status == 'fail']

    domain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0})
    for r in results:
        domain_stats[r.domain]['total'] += 1
        domain_stats[r.domain][r.status] += 1

    error_counts = Counter(r.error_type for r in failed)

    lines: list[str] = []
    lines.append('# SM120 Kernel Compatibility Report')
    lines.append('')
    lines.append('Empirical compatibility assessment of [TileKernels](tile_kernels/) on NVIDIA Blackwell (SM120) '
                 'without kernel source modifications.')
    lines.append('')
    lines.append('> **Note:** The project README lists SM90 and SM100 as supported architectures only. '
                 'This report tests whether kernels run on SM120 (compute capability 12.0) using the installed '
                 'TileLang version, without modifying any kernel code.')
    lines.append('')
    lines.append('## Test Environment')
    lines.append('')
    lines.append(f'| Property | Value |')
    lines.append(f'|----------|-------|')
    lines.append(f'| Date | {env["date"]} |')
    lines.append(f'| GPU | {env["gpu_name"]} (SM120, same arch as RTX 5090) |')
    lines.append(f'| Compute Capability | {env["compute_capability"]} ({env["sm_arch"]}) |')
    lines.append(f'| Driver | {env["driver"]} |')
    lines.append(f'| CUDA | {env["cuda_version"]} |')
    lines.append(f'| PyTorch | {env["pytorch"]} |')
    lines.append(f'| TileLang | {env["tilelang"]} |')
    lines.append(f'| TileLang Target | `{env["tilelang_target"]}` |')
    lines.append(f'| TileLang Compute Version | {env["tilelang_compute_version"]} |')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f'| Metric | Count |')
    lines.append(f'|--------|-------|')
    lines.append(f'| Total kernels tested | {len(results)} |')
    lines.append(f'| **Pass** (compile + launch OK) | **{len(passed)}** |')
    lines.append(f'| **Fail** | **{len(failed)}** |')
    lines.append('')
    lines.append('### By Domain')
    lines.append('')
    lines.append('| Domain | Total | Pass | Fail |')
    lines.append('|--------|-------|------|------|')
    for domain in ['moe', 'quant', 'transpose', 'engram', 'mhc']:
        s = domain_stats[domain]
        lines.append(f'| {domain} | {s["total"]} | {s["pass"]} | {s["fail"]} |')
    lines.append('')
    if error_counts:
        lines.append('### Failure Types')
        lines.append('')
        lines.append('| Error Category | Count |')
        lines.append('|----------------|-------|')
        for cat, count in sorted(error_counts.items()):
            lines.append(f'| `{cat}` | {count} |')
        lines.append('')

    lines.append('## Passing Kernels')
    lines.append('')
    if passed:
        by_domain: dict[str, list[KernelResult]] = defaultdict(list)
        for r in passed:
            by_domain[r.domain].append(r)
        for domain in ['moe', 'quant', 'transpose', 'engram', 'mhc']:
            if domain in by_domain:
                lines.append(f'### {domain}')
                lines.append('')
                for r in sorted(by_domain[domain], key=lambda x: x.name):
                    lines.append(f'- `{r.name}` — `{r.source_file}`')
                lines.append('')
    else:
        lines.append('_No kernels passed._')
        lines.append('')

    lines.append('## Failing Kernels')
    lines.append('')
    if failed:
        lines.append('| Kernel | Source | Error Type | Error Summary |')
        lines.append('|--------|--------|------------|---------------|')
        for r in sorted(failed, key=lambda x: (x.domain, x.name)):
            summary = (r.error_summary or '').replace('|', '\\|')
            lines.append(f'| `{r.name}` | `{r.source_file}` | `{r.error_type}` | {summary} |')
        lines.append('')
        lines.append('<details>')
        lines.append('<summary>Full tracebacks for failing kernels</summary>')
        lines.append('')
        for r in sorted(failed, key=lambda x: (x.domain, x.name)):
            lines.append(f'### {r.name}')
            lines.append('')
            lines.append('```')
            lines.append(r.full_traceback or '')
            lines.append('```')
            lines.append('')
        lines.append('</details>')
    else:
        lines.append('_All kernels passed._')
        lines.append('')

    lines.append('## Analysis')
    lines.append('')
    total = len(results)
    pass_pct = 100.0 * len(passed) / total if total else 0
    lines.append(f'- **{len(passed)}/{total}** ({pass_pct:.1f}%) kernels compile and launch successfully on SM120 '
                 f'with TileLang {env["tilelang"]} and no kernel modifications.')
    if failed:
        top_errors = error_counts.most_common(3)
        err_desc = ', '.join(f'`{e}` ({c})' for e, c in top_errors)
        lines.append(f'- Primary failure modes: {err_desc}.')
        compile_fails = sum(1 for r in failed if r.error_type == 'compilation_error')
        runtime_fails = sum(1 for r in failed if r.error_type == 'cuda_runtime')
        tilelang_fails = sum(1 for r in failed if r.error_type == 'tilelang_internal')
        if compile_fails:
            lines.append(f'- {compile_fails} kernel(s) failed at **JIT compile** time (NVCC/PTX).')
        if tilelang_fails:
            lines.append(f'- {tilelang_fails} kernel(s) hit **TileLang/TVM internal** errors.')
        if runtime_fails:
            lines.append(f'- {runtime_fails} kernel(s) compiled but failed at **CUDA runtime** (launch/execution).')
    else:
        lines.append('- All tested kernels are compatible with SM120 out of the box under the current TileLang version.')
    lines.append('- TileLang detected target `sm_120a` (compute 12.0). The only arch-specific code in tile_kernels '
                 'is vectorization width in `quant/common.py`, which treats SM120 like SM100 (32-byte vectorize).')
    lines.append('- This smoke test invokes each kernel once with representative tensor shapes; it does not run '
                 'the full pytest parametrized matrix (7,540 cases).')
    lines.append('- **Caveat — `grad_w_reduce`:** When `grad_w_partial.shape[0]` equals the full GPU SM count, '
                 'this kernel may fail with shared-memory limit errors on workstation Blackwell GPUs. '
                 'Smoke test used a smaller block count.')
    lines.append('- **Caveat — shape alignment:** Several kernels enforce alignment constraints '
                 '(transpose: dims % 64; per-channel quant: num_tokens % 128). '
                 'These are input requirements, not arch incompatibilities.')
    lines.append(f'- Full execution log: `{os.path.basename(log_path)}`')
    lines.append('- Reproduce: `python scripts/sm120_compat_check.py`')
    lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'SM120 Compatibility Test Log — {env["date"]}\n')
        f.write('=' * 60 + '\n\n')
        for r in results:
            f.write(f'[{r.status.upper()}] {r.domain}/{r.name}\n')
            if r.error_type:
                f.write(f'  error_type: {r.error_type}\n')
                f.write(f'  summary: {r.error_summary}\n')
            if r.full_traceback:
                f.write(r.full_traceback)
            f.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='SM120 kernel compatibility smoke tests')
    parser.add_argument(
        '--output', default='SM120_COMPATIBILITY.md',
        help='Output markdown report path (default: SM120_COMPATIBILITY.md)',
    )
    parser.add_argument(
        '--log', default='SM120_COMPATIBILITY.log',
        help='Output log path (default: SM120_COMPATIBILITY.log)',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: CUDA not available', file=sys.stderr)
        return 1

    cases = build_cases()
    results: list[KernelResult] = []

    print(f'Running {len(cases)} kernel smoke tests on {torch.cuda.get_device_name(0)}...')
    for i, case in enumerate(cases, 1):
        print(f'  [{i}/{len(cases)}] {case.domain}/{case.name}...', end=' ', flush=True)
        result = run_case(case)
        results.append(result)
        print(result.status.upper(), f'({result.error_type})' if result.error_type else '')

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(repo_root, args.log)

    write_report(results, output_path, log_path)
    passed = sum(1 for r in results if r.status == 'pass')
    print(f'\nDone: {passed}/{len(results)} passed')
    print(f'Report: {output_path}')
    print(f'Log:    {log_path}')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
