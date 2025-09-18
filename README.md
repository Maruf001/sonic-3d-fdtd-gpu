## Repository name

**`sonic-3d-fdtd-gpu`**

---

### `README.md`

# sonic-3d-fdtd-gpu
GPU-accelerated 3D acoustic FDTD (finite-difference time-domain) solver with **two backends**:
- **OpenACC** (portable accelerator directives)
- **CUDA** (hand-optimized kernels)

The core model solves the scalar acoustic wave equation in 3D on a staggered/regular grid with 2nd-order (demo) time-stepping and 6/7/27-point spatial stencils. The goal is a clear, didactic codebase that starts simple and then layers in practical GPU optimizations.

> **Status**: Minimal working scaffold with both backends and a shared interface; baseline kernels + simple I/O. Ongoing work: higher-order stencils, absorbing boundaries (PML/CPML), multi-GPU halos, mixed precision, autotuning.

---

## Why this repo
- Compare **OpenACC vs CUDA** on the *same* numerical core.
- Document a path from a naïve GPU port → memory-aware optimizations (blocking, shared-mem tiling, coalescing, L2 reuse, async copies).
- Keep the numerics readable and the perf knobs explicit.

---

## Physics & numerics (brief)
We model the scalar acoustic pressure \(p(\mathbf{x},t)\):
\[
\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p + s(\mathbf{x},t),
\]
with constant/variable sound speed \(c(\mathbf{x})\). Discretization:
- Time: 2nd-order leapfrog (demo), \(\;p^{n+1} = 2p^n - p^{n-1} + \Delta t^2\, c^2 \nabla^2 p^n + \Delta t^2 s^n\)
- Space: 7-point Laplacian stencil (demo). Optional 27-pt / higher order upcoming.
- Stability: CFL \( \Delta t \le \frac{1}{c_{\max}}\left(\frac{1}{\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}+\frac{1}{\Delta z^2}}\right)^{1/2} \)

---

## Features implemented so far
- **Common host driver** with interchangeable backends
- **OpenACC** baseline parallel loops + data regions
- **CUDA** baseline kernel (7-point) + pitch-linear layout
- Simple **point source** and **Dirichlet/zero-flux** boundary options
- Minimal binary output per snapshot (`.npy` or raw float32) via the Python runner
- **Config file** for grid, timestep, source, snapshots

### Optimization work already started
- SoA-style linear layout for coalesced reads
- Optional restrict qualifiers, const device loads
- CUDA: block-tiling in XY (demo), Z-sweep to exploit L2
- OpenACC: `kernels`/`parallel` regions and `cache` hints (where supported)

### Near-term roadmap
- CPML/absorbing boundaries
- Shared-memory tiled CUDA kernels with halo staging
- Asynchronous IO/compute overlap
- Mixed precision and fast-math gating
- Multi-GPU domain decomposition (MPI + CUDA-aware)
- Autotuner for block sizes / tile shapes
- Higher-order stencils + dispersion analysis

---

## Getting started

### 1) Clone
```bash
git clone https://github.com/<you>/sonic-3d-fdtd-gpu.git
cd sonic-3d-fdtd-gpu
````

### 2) Build (CMake)

You can build either backend with flags.

**CUDA backend**

```bash
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build -j
```

**OpenACC backend (NVIDIA HPC SDK)**

```bash
# Ensure NVHPC is loaded, e.g., export CC=nvc CXX=nvc++ FC=nvfortran
cmake -S . -B build -DUSE_OPENACC=ON
cmake --build build -j
```

This produces:

* `build/bin/sonic_fdtd_cuda` or
* `build/bin/sonic_fdtd_openacc`

### 3) Run a quick example

```bash
# small demo domain and a point source
./build/bin/sonic_fdtd_cuda \
  --nx 128 --ny 128 --nz 128 \
  --dx 0.01 --dy 0.01 --dz 0.01 \
  --dt 2.5e-6 --steps 2000 \
  --c0 343.0 \
  --src_x 64 --src_y 64 --src_z 64 \
  --src_freq 1500 \
  --snapshot_every 200 \
  --out_dir out/cuda_demo

# OpenACC variant
./build/bin/sonic_fdtd_openacc <same flags>
```

Or use the small Python wrapper for configs & visualization:

```bash
python3 python/runner.py --config examples/config.yaml
```

---

## Command line flags

Run with `--help` for the full list. Common ones:

* `--nx --ny --nz` : grid size
* `--dx --dy --dz` : spacings
* `--dt --steps`   : time step (respect CFL) and iterations
* `--c0`           : constant sound speed (m/s)
* `--src_x/y/z`, `--src_freq` : point source location & frequency
* `--snapshot_every` : dump cadence
* `--out_dir`      : output path

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
├─ CMakeLists.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ examples/
│  ├─ config.yaml
│  └─ run.sh
├─ python/
│  └─ runner.py
├─ src/
│  ├─ common/
│  │  ├─ args.hpp
│  │  └─ core.hpp
│  ├─ cuda/
│  │  ├─ main.cu
│  │  └─ step.cu
│  └─ openacc/
│     ├─ main.c
│     └─ step.h
└─ docs/
   └─ ROADMAP.md
```

---

## What’s been done (effort summary)

* Baseline **OpenACC** port from CPU loops (parallel regions + persistent data)
* Handwritten **CUDA** kernel with reasonable blocking & coalescing
* Modest **numerical hygiene** (CFL checks, optional symmetry test)
* **I/O**: snapshot dumps for quick sanity plots
* Clear **separation** of numerics (stencil) from backend plumbing

---

## Contributing

Issues and PRs welcome—especially for PML, higher-order stencils, and performance PRs with repro harnesses. See `docs/ROADMAP.md`.

## License

MIT (see `LICENSE`).

## Citation

If this helps your research/product, please cite (see `CITATION.cff`), or star the repo to help others find it 🙏.

