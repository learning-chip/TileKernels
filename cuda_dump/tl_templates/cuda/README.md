# Vendored TileLang CUDA template headers

These headers are copied from TileLang's `src/tl_templates/cuda/` tree so
the dumped kernels under `cuda_dump/tile_kernels/` can resolve
`#include <tl_templates/cuda/...>` locally.

Regenerate with:

```bash
python scripts/copy_tl_templates_headers.py
```

Compile a dumped kernel with:

```bash
nvcc -I cuda_dump -arch=sm_90 your_kernel.cu
```

External dependencies still required at compile time:

- CUDA Toolkit headers (`cuda_runtime.h`, `cuda_fp8.h`, etc.)
- CuTe / CUTLASS (used by GEMM and MMA template headers)

Files vendored (28):

- `atomic.h`
- `barrier.h`
- `common.h`
- `copy.h`
- `copy_sm100.h`
- `copy_sm90.h`
- `cuda_bf16_fallbacks.cuh`
- `cuda_bf16_wrapper.h`
- `cuda_fp4.h`
- `cuda_fp8.h`
- `debug.h`
- `gemm.h`
- `gemm_mma.h`
- `gemm_sm100.h`
- `gemm_sm120.h`
- `gemm_sm70.h`
- `gemm_sm80.h`
- `gemm_sm89.h`
- `gemm_sm90.h`
- `instruction/mma.h`
- `instruction/wgmma.h`
- `intrin.h`
- `ldsm.h`
- `reduce.h`
- `tcgen_05.h`
- `tcgen_05_ld.h`
- `tcgen_05_st.h`
- `threadblock_swizzle.h`

