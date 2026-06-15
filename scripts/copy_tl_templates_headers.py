#!/usr/bin/env python3
"""Copy tl_templates CUDA headers required by cuda_dump .cu files."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DUMP = ROOT / "cuda_dump"
TL_SRC = Path("/mounted_home/work_code/tilelang/src/tl_templates/cuda")
TL_DST = CUDA_DUMP / "tl_templates" / "cuda"

INCLUDE_RE = re.compile(r'#include\s+["<]([^">]+)[">]')


def normalize_rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def resolve_include(inc: str, base_dir: Path, root: Path) -> Path | None:
    inc = inc.strip()
    if inc.startswith("tl_templates/cuda/"):
        return root / inc[len("tl_templates/cuda/") :]
    if inc.startswith("./"):
        inc = inc[2:]
    for candidate in (base_dir / inc, base_dir / Path(inc).name):
        if candidate.exists():
            return candidate.resolve()
    return None


def direct_includes_from_cu() -> set[str]:
    pat = re.compile(r"#include\s+<tl_templates/cuda/([^>]+)>")
    found: set[str] = set()
    for cu in CUDA_DUMP.rglob("*.cu"):
        for match in pat.finditer(cu.read_text()):
            found.add(match.group(1))
    return found


def transitive_closure(seeds: set[str]) -> set[str]:
    queue = sorted(seeds)
    seen = set(queue)
    while queue:
        rel = queue.pop(0)
        src = TL_SRC / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing tl_templates header: {rel}")
        base = src.parent
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line.startswith("#include"):
                continue
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            inc = match.group(1)
            resolved = resolve_include(inc, base, TL_SRC)
            if resolved is None:
                continue
            nrel = normalize_rel(resolved, TL_SRC)
            if nrel not in seen:
                seen.add(nrel)
                queue.append(nrel)
    return seen


def copy_headers(headers: set[str]) -> list[str]:
    copied: list[str] = []
    for rel in sorted(headers):
        src = TL_SRC / rel
        dst = TL_DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def write_readme(headers: list[str]) -> None:
    readme = TL_DST / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Vendored TileLang CUDA template headers",
                "",
                "These headers are copied from TileLang's `src/tl_templates/cuda/` tree so",
                "the dumped kernels under `cuda_dump/tile_kernels/` can resolve",
                "`#include <tl_templates/cuda/...>` locally.",
                "",
                "Regenerate with:",
                "",
                "```bash",
                "python scripts/copy_tl_templates_headers.py",
                "```",
                "",
                "Compile a dumped kernel with:",
                "",
                "```bash",
                "nvcc -I cuda_dump -arch=sm_90 your_kernel.cu",
                "```",
                "",
                "External dependencies still required at compile time:",
                "",
                "- CUDA Toolkit headers (`cuda_runtime.h`, `cuda_fp8.h`, etc.)",
                "- CuTe / CUTLASS (used by GEMM and MMA template headers)",
                "",
                f"Files vendored ({len(headers)}):",
                "",
                *[f"- `{name}`" for name in headers],
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    if not TL_SRC.is_dir():
        raise SystemExit(f"TileLang template source not found: {TL_SRC}")
    seeds = direct_includes_from_cu()
    headers = transitive_closure(seeds)
    copied = copy_headers(headers)
    write_readme(copied)
    print(f"Copied {len(copied)} headers to {TL_DST}")


if __name__ == "__main__":
    main()
