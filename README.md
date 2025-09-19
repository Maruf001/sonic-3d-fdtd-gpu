# sonic-3d-fdtd-gpu
GPU-accelerated 3D acoustic FDTD (finite-difference time-domain) solver with **three implementations**:
- **OpenACC** (portable accelerator directives) - by Dr. Ziyi Yin (KronosAI)
- **CUDA** (hand-optimized kernels) - by Abdullah Maruf
- **Mixed Precision CUDA** (FP16/FP32 variants) - performance-optimized versions

The core model solves the scalar acoustic wave equation in 3D on a staggered/regular grid with 2nd-order time-stepping and spatial stencils. Each implementation includes performance benchmarks and analysis on A100 GPU hardware.

> **Status**: Complete working implementations with performance analysis results. Includes timing comparisons, memory usage profiling, and visualization outputs across different grid sizes (64³ to 1024³).

---

## Why this repo
- Compare **OpenACC vs CUDA vs Mixed Precision** implementations on the same numerical core
- Performance analysis across different grid sizes with comprehensive benchmarking
- Document GPU optimization strategies including memory-aware optimizations and mixed precision techniques
- Practical performance results from A100 GPU testing

---

## Physics & numerics

**Model (scalar acoustic pressure `p(x,t)`):**

```
∂²p/∂t² = c(x)² · ∇²p + s(x,t)
```

* `p` : acoustic pressure
* `c(x)` : sound speed (can be constant or spatially varying)
* `s(x,t)` : source term

**Time discretization (2nd-order leapfrog, demo):**

```
p^{n+1} = 2·p^n − p^{n−1} + (Δt)² [ c² · (∇² p^n) + s^n ]
```

**Space discretization (7-point Laplacian stencil, demo):**

```
∇² p[i,j,k] ≈ ( p[i+1,j,k] − 2·p[i,j,k] + p[i−1,j,k] ) / Δx²
            + ( p[i,j+1,k] − 2·p[i,j,k] + p[i,j−1,k] ) / Δy²
            + ( p[i,j,k+1] − 2·p[i,j,k] + p[i,j,k−1] ) / Δz²
```

> Optional: 27-point / higher-order stencils coming soon.

**CFL stability (Cartesian grid):**

```
Δt ≤ (1 / c_max) · sqrt( 1 / ( 1/Δx² + 1/Δy² + 1/Δz² ) )
```

* `c_max` is the maximum sound speed in the domain.
* Use a safety factor (e.g., 0.9× the bound) in practice.

---

## Implementations

### 1. OpenACC Implementation (`1_pde-fdtd_openacc/`)
- **Author**: Dr. Ziyi Yin (KronosAI)
- Portable accelerator directives implementation
- Performance results available for grid sizes 64³ to 1024³
- Memory usage: 19 MiB (64³) to 17.2 GiB (1024³)
- Execution time: 4ms (64³) to 2.6s (1024³)

### 2. CUDA Implementation (`2_pde-fdtd_cuda/`)
- **Author**: Abdullah Maruf
- Hand-optimized CUDA kernels with basic and shared memory variants
- Tested on A100 GPU, v6e1 TPU, L4 GPU (A100 gives highest performance)
- Includes comprehensive performance analysis and visualization
- Multiple kernel variants for performance comparison

### 3. Mixed Precision CUDA (`3_pde-fdtd_mixed/`)
- **Author**: Abdullah Maruf
- Three variants: v1, v2, v3 with different optimization strategies
- FP16/FP32 mixed precision implementations
- Advanced performance analysis including precision impact studies
- Optimized for A100 GPU architecture

## Performance Results
Each implementation includes:
- **Timing benchmarks** across multiple grid sizes
- **Memory usage profiling** 
- **Performance visualizations** (PNG plots)
- **Comparative analysis** between basic and optimized variants

---

## Getting started

### Running the Implementations

Each implementation is self-contained in its respective directory:

**1. OpenACC Implementation**
```bash
cd 1_pde-fdtd_openacc/
# Compile
bash 2_compile_openacc.sh
# Run experiments
bash 3i_exp1_openacc.sh
bash 3ii_exp2_openacc.sh
# Analyze results
python 4_analysis_openacc.py
```

**2. CUDA Implementation** 
```bash
cd 2_pde-fdtd_cuda/
# Compile
bash 2_compile_cuda.sh
# Run experiments
bash 3i_exp1_cuda.sh
bash 3ii_exp2_cuda.sh
# Analyze results
python 4i_analysis1_cuda.py
python 4ii_analysis2_cuda.py
```

**3. Mixed Precision CUDA**
```bash
cd 3_pde-fdtd_mixed/
# Compile
bash 2_compile_cuda-mixed.sh
# Run experiments
bash 3i_exp1_mixed.sh
# Analyze results
python 4i_analysis1_mixed.py
python 4ii_analysis2_mixed.py
python 4iii_analysis2_mixed-fp16.py
```

### Results
Performance results and visualizations are automatically generated in the `results_gpu-a100/` subdirectories.

---

## Implementation Details

Each implementation follows a similar structure:

* **Source code**: Main FDTD implementation file
* **Compilation script**: Automated build process
* **Experiment scripts**: Standardized performance testing
* **Analysis scripts**: Result processing and visualization

### Grid Sizes Tested
Performance benchmarks are conducted across five grid sizes:
- 64³ (small)
- 128³ (medium-small) 
- 256³ (medium)
- 512³ (large)
- 1024³ (very large)

---

## Performance tips (quick)

* Prefer **power-of-two** friendly block sizes (e.g., `32x8`, `16x16`) and sweep in Z.
* Start from **`-O3 -DNDEBUG`**; gate `--use_fast_math` (CUDA) to measure error vs speed.
* Keep **`dt`** near CFL bound to reduce steps without instability.
* For larger grids, enable **async copy/compute** and experiment with **L2-friendly** depth.
* OpenACC: try `#pragma acc cache` / `tile` clauses where your compiler supports them.

---

## Repository layout

```
sonic-3d-fdtd-gpu/
├─ README.md
├─ LICENSE
├─ 1_pde-fdtd_openacc/
│  ├─ 1_pde-fdtd_openacc_by_ziyi-yin.cpp
│  ├─ 2_compile_openacc.sh
│  ├─ 3i_exp1_openacc.sh
│  ├─ 3ii_exp2_openacc.sh
│  ├─ 4_analysis_openacc.py
│  └─ results_gpu-a100/
├─ 2_pde-fdtd_cuda/
│  ├─ 1_fdtd_cuda_by_abdullah.cu.cpp
│  ├─ 2_compile_cuda.sh
│  ├─ 3i_exp1_cuda.sh
│  ├─ 3ii_exp2_cuda.sh
│  ├─ 4i_analysis1_cuda.py
│  ├─ 4ii_analysis2_cuda.py
│  └─ results_gpu-a100/
├─ 3_pde-fdtd_mixed/
│  ├─ 1i_fdtd_cuda-mixed_v1.cu.cpp
│  ├─ 1ii_fdtd_cuda-mixed_v2.cu.cpp
│  ├─ 1iii_fdtd_cuda-mixed_v3.cu.cpp
│  ├─ 2_compile_cuda-mixed.sh
│  ├─ 3i_exp1_mixed.sh
│  ├─ 4i_analysis1_mixed.py
│  ├─ 4ii_analysis2_mixed.py
│  ├─ 4iii_analysis2_mixed-fp16.py
│  └─ results_gpu-a100/
└─ env/
```

---

## What's been accomplished

* **Three complete implementations** with performance benchmarking:
  - OpenACC implementation by Dr. Ziyi Yin (KronosAI)
  - CUDA implementation by Abdullah Maruf  
  - Mixed precision CUDA variants (v1, v2, v3) by Abdullah Maruf
* **Comprehensive performance analysis** on A100 GPU hardware
* **Automated benchmarking** across multiple grid sizes (64³ to 1024³)
* **Memory usage profiling** and timing analysis
* **Performance visualizations** and comparative studies
* **Mixed precision optimization** studies (FP16/FP32)

---

## Authors & Contributors

- **Dr. Ziyi Yin** (KronosAI) - OpenACC implementation
- **Abdullah Maruf** - CUDA implementation with A100/TPU/L4 testing and Mixed Precision FP16/FP32 optimization variants

## Contributing

Issues and PRs welcome for additional optimizations, new architectures, or extended analysis.

## License

MIT (see `LICENSE`).

## Citation

If this helps your research/product, please cite (see `CITATION.cff`), or star the repo to help others find it 🙏.

