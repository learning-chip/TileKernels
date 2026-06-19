#!/usr/bin/env python3
"""Dump TileLang CUDA C sources for all TileKernels @tilelang.jit factories."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "cuda_dump"
KERNELS_ROOT = ROOT / "tile_kernels"

os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

import torch
from tilelang import language as T
from tilelang.jit import JITImpl

from tile_kernels.config import get_device_num_sms
from tile_kernels.quant.common import CastInputConfig, get_cast_output_config

# Canonical compile-time shapes from tests/testing/generator.py defaults.
HIDDEN = 7168
MHC_MULT = 4
MHC_MULT3 = MHC_MULT * (2 + MHC_MULT)
MHC_HIDDEN = MHC_MULT * HIDDEN
NUM_SMS = get_device_num_sms()
NUM_TOPK = 6
NUM_EXPERTS = 72
NUM_GROUPS = 8
ALIGNMENT = 128


def _bf16_in() -> CastInputConfig:
    return CastInputConfig(torch_dtype=torch.bfloat16, with_sf=False)


def _bf16_in_with_sf() -> CastInputConfig:
    return CastInputConfig(torch_dtype=torch.bfloat16, with_sf=True, sf_block=(128, 128))


def _fp8_in_with_sf() -> CastInputConfig:
    return CastInputConfig(torch_dtype=torch.float8_e4m3fn, with_sf=True, sf_block=(128, 128))


def _e4m3_out_per_token() -> Any:
    return get_cast_output_config("e4m3", (1, 128), round_sf=True)


def _e4m3_out_per_channel() -> Any:
    return get_cast_output_config("e4m3", (128, 1), round_sf=True)


def _e4m3_out_per_block() -> Any:
    return get_cast_output_config("e4m3", (128, 128), round_sf=True)


def _e2m1_out_per_block() -> Any:
    return get_cast_output_config("e2m1", (128, 128), round_sf=True)


def _e5m6_out() -> Any:
    return get_cast_output_config("e5m6", (1, 128), round_sf=True)


def _e5m6_in() -> CastInputConfig:
    return CastInputConfig(torch_dtype=torch.uint32, with_sf=True, sf_block=(128, 128))


# Registry: module import path -> factory name -> compile spec.
# Each spec: {"args": [...], "kwargs": {...}, "public_api": "..."}
CANONICAL_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "tile_kernels.quant.swiglu_forward_and_per_token_cast_kernel": {
        "get_swiglu_forward_and_per_token_cast_kernel": {
            "args": [HIDDEN // 2, False, False, False, False, T.bfloat16, _e4m3_out_per_token(), None],
            "public_api": "tile_kernels.quant.swiglu_forward_and_per_token_cast",
        },
    },
    "tile_kernels.quant.swiglu_backward_and_per_token_cast_kernel": {
        "get_swiglu_backward_and_per_token_cast_kernel": {
            "args": [HIDDEN // 2, _e4m3_out_per_token(), False],
            "public_api": "tile_kernels.quant.swiglu_backward_and_per_token_cast",
        },
    },
    "tile_kernels.quant.per_token_cast_kernel": {
        "get_per_token_cast_kernel": {
            "args": [HIDDEN, HIDDEN, _bf16_in(), _e4m3_out_per_token()],
            "public_api": "tile_kernels.quant.per_token_cast",
        },
    },
    "tile_kernels.quant.cast_back_kernel": {
        "get_cast_back_kernel": {
            "args": [HIDDEN, _fp8_in_with_sf(), T.bfloat16],
            "public_api": "tile_kernels.quant.cast_back",
        },
    },
    "tile_kernels.quant.per_block_cast_kernel": {
        "get_per_block_cast_kernel": [
            {
                "variant": "e4m3",
                "args": [HIDDEN, _bf16_in(), _e4m3_out_per_block()],
                "public_api": "tile_kernels.quant.per_block_cast",
            },
            {
                "variant": "e2m1",
                "args": [HIDDEN, _bf16_in(), _e2m1_out_per_block()],
                "public_api": "tile_kernels.quant.per_block_cast",
            },
        ],
    },
    "tile_kernels.quant.per_channel_cast_fused_kernel": {
        "get_per_channel_cast_fused_kernel": {
            "args": [HIDDEN, False, _bf16_in(), _e4m3_out_per_channel()],
            "public_api": "tile_kernels.quant.per_channel_cast_fused",
        },
    },
    "tile_kernels.quant.per_block_cast_lossless_kernel": {
        "get_per_block_cast_lossless_kernel": {
            "args": [
                HIDDEN,
                HIDDEN,
                CastInputConfig(torch_dtype=torch.int8, with_sf=True, sf_block=(1, 32)),
                get_cast_output_config("e4m3", (128, 128), round_sf=True),
            ],
            "public_api": "tile_kernels.quant.per_block_cast_lossless",
        },
    },
    "tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose_kernel": {
        "get_swiglu_forward_and_per_channel_cast_and_transpose_kernel": {
            "args": [HIDDEN, False, False, T.bfloat16, _e4m3_out_per_channel(), 0.0],
            "public_api": "tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose",
        },
    },
    "tile_kernels.quant.per_channel_cast_and_transpose_kernel": {
        "get_per_channel_cast_and_transpose_kernel": {
            "args": [HIDDEN, T.bfloat16, _e4m3_out_per_channel()],
            "public_api": "tile_kernels.quant.per_channel_cast_and_transpose",
        },
    },
    "tile_kernels.quant.cast_back_e5m6_kernel": {
        "get_cast_back_e5m6_kernel": {
            "args": [HIDDEN, _e5m6_in(), T.bfloat16],
            "public_api": "tile_kernels.quant.cast_back_e5m6",
        },
    },
    "tile_kernels.quant.per_token_cast_to_e5m6_kernel": {
        "get_per_token_cast_to_e5m6_kernel": {
            "args": [
                HIDDEN,
                HIDDEN,
                _bf16_in(),
                get_cast_output_config("e5m6", (1, HIDDEN), round_sf=True, custom_clamp_min_value=1e-4),
            ],
            "public_api": "tile_kernels.quant.per_token_cast_to_e5m6",
        },
    },
    "tile_kernels.transpose.batched_transpose_kernel": {
        "get_batched_transpose_kernel": {
            "args": [0, 0, T.bfloat16],
            "public_api": "tile_kernels.transpose.transpose",
        },
    },
    "tile_kernels.moe.aux_fi_kernel": {
        "get_aux_fi_kernel": {
            "args": [NUM_TOPK, NUM_EXPERTS, NUM_SMS],
            "public_api": "tile_kernels.moe.aux_fi",
        },
    },
    "tile_kernels.moe.expand_to_fused_kernel": {
        "get_expand_to_fused_kernel": {
            "args": [HIDDEN, NUM_TOPK, 128, False, False, T.float8_e4m3, T.float32],
            "public_api": "tile_kernels.moe.expand_to_fused_with_sf",
        },
    },
    "tile_kernels.moe.get_fused_mapping_kernel": {
        "get_get_fused_mapping_kernel": {
            "args": [NUM_EXPERTS, NUM_TOPK, ALIGNMENT, NUM_SMS],
            "public_api": "tile_kernels.moe.get_fused_mapping",
        },
    },
    "tile_kernels.moe.topk_gate_kernel": {
        "get_topk_gate_kernel": {
            "args": [NUM_EXPERTS, NUM_TOPK],
            "public_api": "tile_kernels.moe.topk_gate",
        },
    },
    "tile_kernels.moe.mask_indices_by_tp_kernel": {
        "get_mask_indices_by_tp_kernel": {
            "args": [NUM_TOPK, T.int64],
            "public_api": "tile_kernels.moe.mask_indices_by_tp",
        },
    },
    "tile_kernels.moe.normalize_weight_kernel": {
        "get_normalize_weight_kernel": {
            "args": [NUM_TOPK],
            "public_api": "tile_kernels.moe.normalize_weight",
        },
    },
    "tile_kernels.moe.inplace_unique_group_indices_kernel": {
        "get_inplace_unique_group_indices_kernel": {
            "args": [NUM_TOPK, 64, NUM_SMS],
            "public_api": "tile_kernels.moe.inplace_unique_group_indices",
        },
    },
    "tile_kernels.moe.group_count_kernel": {
        "get_group_count_kernel": {
            "args": [NUM_TOPK, NUM_GROUPS, NUM_SMS],
            "public_api": "tile_kernels.moe.group_count",
        },
    },
    "tile_kernels.moe.top2_sum_gate_kernel": {
        "get_top2_sum_gate_kernel": {
            "args": [0, NUM_TOPK, 8, 8, 256, False, False, False, False],
            "public_api": "tile_kernels.moe.top2_sum_gate",
        },
    },
    "tile_kernels.moe.reduce_fused_kernel": {
        "get_reduce_fused_kernel": {
            "args": [HIDDEN, NUM_TOPK, T.bfloat16, T.bfloat16, False, True, False],
            "public_api": "tile_kernels.moe.reduce_fused",
        },
    },
    "tile_kernels.moe.topk_sum_and_topk_group_idx_kernel": {
        "get_topk_sum_and_topk_group_idx_kernel": {
            "args": [8, 32, 8, 2],
            "public_api": "tile_kernels.moe.topk_sum_and_topk_group_idx",
        },
    },
    "tile_kernels.engram.engram_hash_kernel": {
        "get_engram_hash_kernel": {
            "args": [],
            "kwargs": {"max_ngram_size": 3, "num_ngram_layers": 2, "num_embed_table_per_ngram": 8},
            "public_api": "tile_kernels.engram.engram_hash",
        },
    },
    "tile_kernels.engram.engram_grad_w_reduce_kernel": {
        "get_engram_grad_w_reduce_kernel": {
            "args": [HIDDEN, NUM_SMS, 4],
            "public_api": "tile_kernels.engram.grad_w_reduce",
        },
    },
    "tile_kernels.engram.engram_gate_kernel": {
        "get_engram_gate_fwd_kernel": {
            "args": [
                HIDDEN,
                1e-20,
                HIDDEN**-0.5,
                MHC_MULT * HIDDEN,
                HIDDEN,
                HIDDEN,
                NUM_SMS,
            ],
            "kwargs": {"clamp_value": 1e-6, "hc_mult": 4, "save_for_backward": True},
            "public_api": "tile_kernels.engram.engram_gate_fwd",
        },
        "get_engram_gate_bwd_kernel": {
            "args": [HIDDEN, HIDDEN**-0.5, MHC_MULT * HIDDEN, HIDDEN, HIDDEN, NUM_SMS],
            "kwargs": {"clamp_value": 1e-6, "hc_mult": 4},
            "public_api": "tile_kernels.engram.engram_gate_bwd",
        },
    },
    "tile_kernels.engram.engram_fused_weight_kernel": {
        "get_engram_fused_weight_kernel": {
            "args": [HIDDEN, 4],
            "public_api": "tile_kernels.engram.fused_weight",
        },
    },
    "tile_kernels.mhc.norm_fn_kernel": {
        "_mhc_fn_normw_merge_fwd": {
            "args": [4096, MHC_HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_fn_normw_merge",
        },
        "_mhc_fn_normw_merge_bwd": {
            "args": [4096, MHC_HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_fn_normw_merge",
        },
        "_mhc_pre_norm_fn_fwd_mul": {
            "args": [MHC_MULT3, 1, MHC_HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_pre_norm_fn",
        },
        "_mhc_pre_norm_fn_fwd_norm": {
            "args": [MHC_MULT3, 1, MHC_HIDDEN, 1e-6, 1],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_pre_norm_fn",
        },
        "_mhc_pre_norm_fn_bwd_norm": {
            "args": [MHC_MULT3, 1, MHC_HIDDEN, 1e-6],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_pre_norm_fn",
        },
        "_mhc_pre_norm_fn_bwd_mul": {
            "args": [MHC_MULT3, 1, MHC_HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.norm_fn.mhc_pre_norm_fn",
        },
    },
    "tile_kernels.mhc.post_kernel": {
        "_mhc_post_fwd": {
            "args": [MHC_MULT, HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.post.mhc_post",
        },
        "_mhc_post_bwd": {
            "args": [MHC_MULT, HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.post.mhc_post",
        },
    },
    "tile_kernels.mhc.expand_kernel": {
        "expand_to_mhc_fwd_tl": {
            "args": [HIDDEN, MHC_MULT],
            "public_api": "tile_kernels.modeling.mhc.ops.expand.expand_to_mhc",
        },
        "expand_to_mhc_bwd_tl": {
            "args": [HIDDEN, MHC_MULT],
            "public_api": "tile_kernels.modeling.mhc.ops.expand.expand_to_mhc",
        },
    },
    "tile_kernels.mhc.head_compute_mix_kernel": {
        "_mhc_head_compute_mix_fwd": {
            "args": [MHC_MULT, 1e-6, 32],
            "public_api": "tile_kernels.modeling.mhc.ops.head_compute_mix.mhc_head_compute_mix",
        },
        "_mhc_head_compute_mix_bwd": {
            "args": [MHC_MULT, 32, NUM_SMS],
            "public_api": "tile_kernels.modeling.mhc.ops.head_compute_mix.mhc_head_compute_mix",
        },
    },
    "tile_kernels.mhc.sinkhorn_kernel": {
        "_mhc_sinkhorn_fwd": {
            "args": [MHC_MULT, 1, 10, 1e-6],
            "public_api": "tile_kernels.modeling.mhc.ops.sinkhorn.sinkhorn_normalize",
        },
        "_mhc_sinkhorn_bwd": {
            "args": [MHC_MULT, 32, 10, 1e-6],
            "public_api": "tile_kernels.modeling.mhc.ops.sinkhorn.sinkhorn_normalize",
        },
    },
    "tile_kernels.mhc.pre_apply_mix_kernel": {
        "_mhc_pre_apply_mix_fwd": {
            "args": [MHC_MULT, HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.pre_apply_mix.mhc_pre_apply_mix",
        },
        "_mhc_pre_apply_mix_bwd": {
            "args": [MHC_MULT, HIDDEN],
            "public_api": "tile_kernels.modeling.mhc.ops.pre_apply_mix.mhc_pre_apply_mix",
        },
    },
    "tile_kernels.mhc.pre_split_mixes_kernel": {
        "_mhc_pre_split_mixes_fwd": {
            "args": [MHC_MULT, 1.0, 1e-6, 32],
            "public_api": "tile_kernels.modeling.mhc.ops.pre_split_mixes.mhc_pre_split_mixes",
        },
        "_mhc_pre_split_mixes_bwd": {
            "args": [MHC_MULT, 1.0, 32, NUM_SMS],
            "public_api": "tile_kernels.modeling.mhc.ops.pre_split_mixes.mhc_pre_split_mixes",
        },
    },
    "tile_kernels.mhc.pre_big_fuse_kernel": {
        "_mhc_pre_big_fuse": {
            "args": [HIDDEN, 1e-6, 1e-6, 1e-6, 1.0, 10],
            "public_api": "tile_kernels.modeling.mhc.ops.pre_big_fuse.mhc_pre_big_fuse",
        },
    },
    "tile_kernels.mhc.multilayer_recompute_kernel": {
        "_mhc_multilayer_recompute_kernel": {
            "args": [MHC_MULT, HIDDEN, 10, 9],
            "public_api": "tile_kernels.modeling.mhc.ops.multilayer_recompute.mhc_multilayer_recompute",
        },
    },
}


def count_jit_decorators(py_file: Path) -> int:
    tree = ast.parse(py_file.read_text())
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                count += 1
                break
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "jit":
                count += 1
                break
    return count


def module_import_path(py_file: Path) -> str:
    rel = py_file.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


EXPECTED_DUMP_COUNT = 49


def normalize_factory_specs(spec: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(spec, list):
        return spec
    return [spec]


def cuda_output_path(
    py_file: Path,
    factory_name: str,
    multi_jit: bool,
    variant: str | None = None,
) -> Path:
    rel = py_file.relative_to(KERNELS_ROOT)
    stem = rel.with_suffix("")
    if multi_jit:
        return OUTPUT_ROOT / "tile_kernels" / f"{stem}__{factory_name}.cu"
    if variant:
        return OUTPUT_ROOT / "tile_kernels" / f"{stem}__{variant}.cu"
    return OUTPUT_ROOT / "tile_kernels" / f"{stem}.cu"


def serialize_value(value: Any) -> Any:
    if isinstance(value, T.dtype):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if is_dataclass(value):
        return {k: serialize_value(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    return value


def format_header(
    source_py: str,
    factory_name: str,
    prim_func: str,
    compile_args: dict[str, Any],
    public_api: str | None,
    variant: str | None = None,
) -> str:
    lines = [
        "// TileKernels CUDA dump (generated by scripts/dump_cuda_sources.py)",
        f"// Source Python: {source_py}",
        f"// JIT factory: {factory_name}",
        f"// prim_func: {prim_func}",
    ]
    if variant:
        lines.append(f"// Variant: {variant}")
    lines.append(f"// Compile args: {json.dumps(serialize_value(compile_args), sort_keys=True)}")
    if public_api:
        lines.append(f"// Public API: {public_api}")
    lines.append("")
    return "\n".join(lines)


def discover_kernel_modules() -> list[Path]:
    return sorted(KERNELS_ROOT.rglob("*_kernel.py"))


def discover_jit_factories(module) -> list[tuple[str, JITImpl]]:
    factories: list[tuple[str, JITImpl]] = []
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, JITImpl):
            factories.append((name, obj))
    return sorted(factories, key=lambda item: item[0])


def compile_and_dump(
    factory: JITImpl,
    args: list[Any],
    kwargs: dict[str, Any],
) -> tuple[str, str]:
    compiled = factory(*args, **kwargs)
    source = compiled.get_kernel_source()
    prim_func = compiled.prim_func.attrs.get("global_symbol", "unknown")
    return source, str(prim_func)


def write_manifest_md(entries: list[dict[str, Any]]) -> None:
    path = OUTPUT_ROOT / "MANIFEST.md"
    lines = [
        "# TileKernels CUDA Dump Manifest",
        "",
        "| CUDA file | Source Python | JIT factory | prim_func | Public API |",
        "|---|---|---|---|---|",
    ]
    for entry in sorted(entries, key=lambda e: e["cuda_file"]):
        lines.append(
            f"| `{entry['cuda_file']}` | `{entry['source_py']}` | `{entry['jit_factory']}` | "
            f"`{entry['prim_func']}` | `{entry.get('public_api', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required to compile TileLang kernels.", file=sys.stderr)
        return 1

    manifest: list[dict[str, Any]] = []
    failures: list[str] = []

    kernel_modules = discover_kernel_modules()
    print(f"Found {len(kernel_modules)} kernel modules")

    for py_file in kernel_modules:
        import_path = module_import_path(py_file)
        rel_py = str(py_file.relative_to(ROOT))
        jit_count = count_jit_decorators(py_file)
        if jit_count == 0:
            print(f"SKIP (no @tilelang.jit): {rel_py}")
            continue

        module = importlib.import_module(import_path)
        factories = discover_jit_factories(module)
        if len(factories) != jit_count:
            print(
                f"WARN: {rel_py} AST count={jit_count}, discovered={len(factories)} "
                f"({[name for name, _ in factories]})"
            )

        module_specs = CANONICAL_SPECS.get(import_path, {})
        multi_jit = len(factories) > 1

        for factory_name, factory in factories:
            raw_spec = module_specs.get(factory_name)
            if raw_spec is None:
                failures.append(f"{import_path}::{factory_name}: missing canonical compile spec")
                continue

            for spec in normalize_factory_specs(raw_spec):
                variant = spec.get("variant")
                args = spec.get("args", [])
                kwargs = spec.get("kwargs", {})
                compile_args = {
                    "args": serialize_value(args),
                    "kwargs": serialize_value(kwargs),
                }
                if variant:
                    compile_args["variant"] = variant

                try:
                    source, prim_func = compile_and_dump(factory, args, kwargs)
                except Exception:
                    failures.append(f"{import_path}::{factory_name}::{variant or 'default'}:\n{traceback.format_exc()}")
                    continue

                use_variant_suffix = variant is not None and len(normalize_factory_specs(raw_spec)) > 1
                out_path = cuda_output_path(
                    py_file,
                    factory_name,
                    multi_jit,
                    variant if use_variant_suffix else None,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                header = format_header(
                    rel_py,
                    factory_name,
                    prim_func,
                    compile_args,
                    spec.get("public_api"),
                    variant,
                )
                out_path.write_text(header + source)
                cuda_rel = str(out_path.relative_to(OUTPUT_ROOT))
                entry = {
                    "cuda_file": cuda_rel,
                    "source_py": rel_py,
                    "jit_factory": factory_name,
                    "prim_func": prim_func,
                    "compile_args": compile_args,
                    "public_api": spec.get("public_api"),
                }
                if variant:
                    entry["variant"] = variant
                manifest.append(entry)
                print(f"OK  {cuda_rel}")

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    write_manifest_md(manifest)

    print(f"\nDumped {len(manifest)} CUDA files to {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    if len(manifest) != EXPECTED_DUMP_COUNT:
        print(f"WARN: expected {EXPECTED_DUMP_COUNT} dumps, got {len(manifest)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
