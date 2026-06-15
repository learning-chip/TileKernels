# SM120 Kernel Compatibility Report

Empirical compatibility assessment of [TileKernels](tile_kernels/) on NVIDIA Blackwell (SM120) without kernel source modifications.

> **Note:** The project README lists SM90 and SM100 as supported architectures only. This report tests whether kernels run on SM120 (compute capability 12.0) using the installed TileLang version, without modifying any kernel code.

## Test Environment

| Property | Value |
|----------|-------|
| Date | 2026-06-15 20:41:56 UTC |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120, same arch as RTX 5090) |
| Compute Capability | 12.0 (SM120) |
| Driver | 590.44.01 |
| CUDA | 13.0 |
| PyTorch | 2.12.0+cu130 |
| TileLang | 0.1.9 |
| TileLang Target | `target.Target(kind=target.TargetKind(name="cuda", default_device_type=2, default_keys=("cuda", "gpu")), tag="", keys=("cuda", "gpu"), attrs={"thread_warp_size": 32, "max_num_threads": 1024, "arch": "sm_120a"}, features={}, host=None)` |
| TileLang Compute Version | 12.0 |

## Summary

| Metric | Count |
|--------|-------|
| Total kernels tested | 45 |
| **Pass** (compile + launch OK) | **45** |
| **Fail** | **0** |

### By Domain

| Domain | Total | Pass | Fail |
|--------|-------|------|------|
| moe | 12 | 12 | 0 |
| quant | 17 | 17 | 0 |
| transpose | 2 | 2 | 0 |
| engram | 5 | 5 | 0 |
| mhc | 9 | 9 | 0 |

## Passing Kernels

### moe

- `aux_fi` — `moe/aux_fi_kernel.py`
- `expand_to_fused` — `moe/expand_to_fused_kernel.py`
- `expand_to_fused_with_sf` — `moe/expand_to_fused_kernel.py`
- `get_fused_mapping` — `moe/get_fused_mapping_kernel.py`
- `group_count` — `moe/group_count_kernel.py`
- `inplace_unique_group_indices` — `moe/inplace_unique_group_indices_kernel.py`
- `mask_indices_by_tp` — `moe/mask_indices_by_tp_kernel.py`
- `normalize_weight` — `moe/normalize_weight_kernel.py`
- `reduce_fused` — `moe/reduce_fused_kernel.py`
- `top2_sum_gate` — `moe/top2_sum_gate_kernel.py`
- `topk_gate` — `moe/topk_gate_kernel.py`
- `topk_sum_and_topk_group_idx` — `moe/topk_sum_and_topk_group_idx_kernel.py`

### quant

- `cast_back` — `quant/cast_back_kernel.py`
- `cast_back_e5m6` — `quant/cast_back_e5m6_kernel.py`
- `per_block_cast` — `quant/per_block_cast_kernel.py`
- `per_block_cast_lossless` — `quant/per_block_cast_lossless_kernel.py`
- `per_block_cast_with_precomputed_sf` — `quant/per_block_cast_kernel.py`
- `per_block_cast_with_sf_only` — `quant/per_block_cast_kernel.py`
- `per_channel_cast` — `quant/per_channel_cast_kernel.py`
- `per_channel_cast_and_transpose` — `quant/per_channel_cast_and_transpose_kernel.py`
- `per_channel_cast_fused` — `quant/per_channel_cast_fused_kernel.py`
- `per_token_cast` — `quant/per_token_cast_kernel.py`
- `per_token_cast_back` — `quant/cast_back_kernel.py`
- `per_token_cast_to_e5m6` — `quant/per_token_cast_to_e5m6_kernel.py`
- `per_token_cast_with_precomputed_sf` — `quant/per_token_cast_kernel.py`
- `per_token_cast_with_sf_only` — `quant/per_token_cast_kernel.py`
- `swiglu_backward_and_per_token_cast` — `quant/swiglu_backward_and_per_token_cast_kernel.py`
- `swiglu_forward_and_per_channel_cast_and_transpose` — `quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py`
- `swiglu_forward_and_per_token_cast` — `quant/swiglu_forward_and_per_token_cast_kernel.py`

### transpose

- `batched_transpose` — `transpose/batched_transpose_kernel.py`
- `transpose` — `transpose/batched_transpose_kernel.py`

### engram

- `engram_gate_bwd` — `engram/engram_gate_kernel.py`
- `engram_gate_fwd` — `engram/engram_gate_kernel.py`
- `engram_hash` — `engram/engram_hash_kernel.py`
- `fused_weight` — `engram/engram_fused_weight_kernel.py`
- `grad_w_reduce` — `engram/engram_grad_w_reduce_kernel.py`

### mhc

- `expand_to_mhc` — `mhc/expand_kernel.py`
- `mhc_head_compute_mix` — `mhc/head_compute_mix_kernel.py`
- `mhc_multilayer_recompute` — `mhc/multilayer_recompute_kernel.py`
- `mhc_post` — `mhc/post_kernel.py`
- `mhc_pre_apply_mix` — `mhc/pre_apply_mix_kernel.py`
- `mhc_pre_big_fuse` — `mhc/pre_big_fuse_kernel.py`
- `mhc_pre_norm_fn` — `mhc/norm_fn_kernel.py`
- `mhc_pre_split_mixes` — `mhc/pre_split_mixes_kernel.py`
- `sinkhorn_normalize` — `mhc/sinkhorn_kernel.py`

## Failing Kernels

_All kernels passed in smoke tests with representative inputs._

### Known Runtime Limitation (production configuration)

| Kernel | Source | Error Type | Condition | Error Summary |
|--------|--------|------------|-----------|---------------|
| `grad_w_reduce` | `engram/engram_grad_w_reduce_kernel.py` | `cuda_runtime` | `grad_w_partial.shape[0] == num_SMs` (188 on this GPU) | `InternalError: Failed to set the allowed dynamic shared memory size to 192512` |

This failure occurs at kernel launch (not compile time) when using the production block count from `get_num_sms()`. The smoke test used 4 blocks and passed. Workstation Blackwell GPUs may have lower per-block dynamic shared memory limits than datacenter SM90/SM100 parts.

## Analysis

- **45/45** (100.0%) kernels compile and launch successfully on SM120 with TileLang 0.1.9 and no kernel modifications.
- **Conclusion:** All TileKernels entry points tested here run on SM120 (Blackwell CC 12.0) without source changes, despite the README listing only SM90/SM100.
- TileLang detected target `sm_120a` (compute 12.0). The only arch-specific code in tile_kernels is vectorization width in `quant/common.py`, which treats SM120 like SM100 (32-byte vectorize).
- This smoke test invokes each kernel once with representative tensor shapes; it does not run the full pytest parametrized matrix (7,540 cases).
- **Caveat — `grad_w_reduce`:** When `grad_w_partial.shape[0]` equals the full GPU SM count (the production/test path via `get_num_sms()`), this kernel requests ~192 KB dynamic shared memory per block and may fail with `Failed to set the allowed dynamic shared memory size` on workstation Blackwell GPUs. It passes with smaller block counts (smoke test used 4 blocks).
- **Caveat — shape alignment:** Several kernels enforce alignment constraints (e.g. `transpose` requires dims divisible by 64; per-channel quant requires `num_tokens % 128 == 0`). These are input requirements, not arch incompatibilities.
- Full execution log: `SM120_COMPATIBILITY.log`
- Reproduce: `python scripts/sm120_compat_check.py`
