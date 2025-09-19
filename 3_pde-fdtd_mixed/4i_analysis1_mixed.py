# Benchmark analysis

import matplotlib.pyplot as plt
import numpy as np

# Read grid sizes from OpenACC file (consistent across all)
with open('result_openacc.txt', 'r') as f:
    lines_openacc = f.readlines()

grid_sizes = [int(x) for x in lines_openacc[0].split(':')[1].strip().split()]
openacc_times = [float(x) for x in lines_openacc[1].split(':')[1].strip().split()]
openacc_mems = [int(x) for x in lines_openacc[2].split(':')[1].strip().split()]

# Read data from CUDA file (FP32)
with open('result_cuda.txt', 'r') as f:
    lines_cuda = f.readlines()

fp32_basic_times = [float(x) for x in lines_cuda[1].split(':')[1].strip().split()]
fp32_basic_mems = [int(x) for x in lines_cuda[2].split(':')[1].strip().split()]
fp32_shared_times = [float(x) for x in lines_cuda[3].split(':')[1].strip().split()]
fp32_shared_mems = [int(x) for x in lines_cuda[4].split(':')[1].strip().split()]

# Read data from Mixed file (FP16)
with open('result_mixed.txt', 'r') as f:
    lines_mixed = f.readlines()

# Corrected line indices for result_mixed.txt based on Z2w9qOpg9ftR output
fp16_basic_times = [float(x) for x in lines_mixed[0].split(':')[1].strip().split()]
fp16_basic_mems = [int(x) for x in lines_mixed[1].split(':')[1].strip().split()]
fp16_shared_times = [float(x) for x in lines_mixed[2].split(':')[1].strip().split()]
fp16_shared_mems = [int(x) for x in lines_mixed[3].split(':')[1].strip().split()]

# Time comparison plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, openacc_times, label='OpenACC', marker='o')
plt.plot(grid_sizes, fp32_basic_times, label='FP32 Basic', marker='x')
plt.plot(grid_sizes, fp32_shared_times, label='FP32 Shared', marker='^')
plt.plot(grid_sizes, fp16_basic_times, label='FP16 Basic', marker='s')
plt.plot(grid_sizes, fp16_shared_times, label='FP16 Shared', marker='p')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Execution Time (s)')
plt.yscale('log')  # Log scale for better visibility
plt.title('Time Comparison: OpenACC vs FP32 CUDA vs FP16 Mixed (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Memory comparison plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, openacc_mems, label='OpenACC', marker='o')
plt.plot(grid_sizes, fp32_basic_mems, label='FP32 Basic', marker='x')
plt.plot(grid_sizes, fp32_shared_mems, label='FP32 Shared', marker='^')
plt.plot(grid_sizes, fp16_basic_mems, label='FP16 Basic', marker='s')
plt.plot(grid_sizes, fp16_shared_mems, label='FP16 Shared', marker='p')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Peak GPU Memory (MiB)')
plt.title('Memory Comparison: OpenACC vs FP32 CUDA vs FP16 Mixed')
plt.legend()
plt.grid()
plt.show()

# Speedup plot (over OpenACC baseline)
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, [o / b for o, b in zip(openacc_times, fp32_basic_times)], label='FP32 Basic vs OpenACC', marker='x')
plt.plot(grid_sizes, [o / s for o, s in zip(openacc_times, fp32_shared_times)], label='FP32 Shared vs OpenACC', marker='^')
plt.plot(grid_sizes, [o / b for o, b in zip(openacc_times, fp16_basic_times)], label='FP16 Basic vs OpenACC', marker='s')
plt.plot(grid_sizes, [o / s for o, s in zip(openacc_times, fp16_shared_times)], label='FP16 Shared vs OpenACC', marker='p')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Speedup (time_OpenACC / time_variant)')
plt.title('Speedup over OpenACC Baseline')
plt.legend()
plt.grid()
plt.show()