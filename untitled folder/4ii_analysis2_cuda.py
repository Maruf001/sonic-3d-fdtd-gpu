# Benchmark analysis

import matplotlib.pyplot as plt
import numpy as np

# Read data from OpenACC file
with open('result_openacc.txt', 'r') as f:
    lines_openacc = f.readlines()

grid_sizes = [int(x) for x in lines_openacc[0].split(':')[1].strip().split()]
openacc_times = [float(x) for x in lines_openacc[1].split(':')[1].strip().split()]
openacc_mems = [int(x) for x in lines_openacc[2].split(':')[1].strip().split()]

# Read data from CUDA file
with open('result_cuda.txt', 'r') as f:
    lines_cuda = f.readlines()

cuda_basic_times = [float(x) for x in lines_cuda[1].split(':')[1].strip().split()]
cuda_basic_mems = [int(x) for x in lines_cuda[2].split(':')[1].strip().split()]
cuda_shared_times = [float(x) for x in lines_cuda[3].split(':')[1].strip().split()]
cuda_shared_mems = [int(x) for x in lines_cuda[4].split(':')[1].strip().split()]

# 1. Time comparison with log scale for better visibility of differences
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, openacc_times, label='OpenACC', marker='o')
plt.plot(grid_sizes, cuda_basic_times, label='CUDA Basic', marker='x')
plt.plot(grid_sizes, cuda_shared_times, label='CUDA Shared', marker='^')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Execution Time (s)')
plt.yscale('log')  # Log scale to highlight low CUDA times
plt.title('Time Comparison: OpenACC vs CUDA Variants (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# 2. Speedup plot (corrected label for clarity)
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, [o / b for o, b in zip(openacc_times, cuda_basic_times)], label='CUDA Basic vs OpenACC', marker='x')
plt.plot(grid_sizes, [o / s for o, s in zip(openacc_times, cuda_shared_times)], label='CUDA Shared vs OpenACC', marker='^')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Speedup (time_OpenACC / time_CUDA)')
plt.title('CUDA Speedup over OpenACC Baseline')
plt.legend()
plt.grid()
plt.show()

# 3. Bar chart for speedups at each grid size
width = 0.35
x = np.arange(len(grid_sizes))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [o / b for o, b in zip(openacc_times, cuda_basic_times)], width, label='CUDA Basic')
ax.bar(x + width/2, [o / s for o, s in zip(openacc_times, cuda_shared_times)], width, label='CUDA Shared')

ax.set_xlabel('Grid Size (n^3)')
ax.set_ylabel('Speedup (time_OpenACC / time_CUDA)')
ax.set_title('CUDA Speedup over OpenACC by Grid Size')
ax.set_xticks(x)
ax.set_xticklabels(grid_sizes)
ax.legend()
ax.grid(axis='y')
plt.show()

# 4. Combined time and memory in subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Time subplot
axs[0].plot(grid_sizes, openacc_times, label='OpenACC', marker='o')
axs[0].plot(grid_sizes, cuda_basic_times, label='CUDA Basic', marker='x')
axs[0].plot(grid_sizes, cuda_shared_times, label='CUDA Shared', marker='^')
axs[0].set_xlabel('Grid Size (n^3)')
axs[0].set_ylabel('Execution Time (s)')
axs[0].set_title('Time Comparison')
axs[0].legend()
axs[0].grid()

# Memory subplot
axs[1].plot(grid_sizes, openacc_mems, label='OpenACC', marker='o')
axs[1].plot(grid_sizes, cuda_basic_mems, label='CUDA Basic', marker='x')
axs[1].plot(grid_sizes, cuda_shared_mems, label='CUDA Shared', marker='^')
axs[1].set_xlabel('Grid Size (n^3)')
axs[1].set_ylabel('Peak GPU Memory (MiB)')
axs[1].set_title('Memory Comparison')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()