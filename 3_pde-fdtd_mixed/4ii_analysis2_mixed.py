# Benchmark analysis

import matplotlib.pyplot as plt

# Hardcode grid sizes since they are consistent across experiments and not always present in all files
grid_sizes = [64, 128, 256, 512, 1024]

# Read data from CUDA file (FP32 baseline)
with open('result_cuda.txt', 'r') as f:
    lines_cuda = f.readlines()

# Assuming the format in result_cuda.txt is:
# Grid Sizes: ...
# Basic Times: ...
# Basic Memories: ...
# Shared Times: ...
# Shared Memories: ...
# Read time and memory based on line indices
fp32_basic_times = [float(x) for x in lines_cuda[1].split(':')[1].strip().split()]
fp32_basic_mems = [int(x) for x in lines_cuda[2].split(':')[1].strip().split()]
fp32_shared_times = [float(x) for x in lines_cuda[3].split(':')[1].strip().split()]
fp32_shared_mems = [int(x) for x in lines_cuda[4].split(':')[1].strip().split()]

# Read data from Mixed file (FP16)
with open('result_mixed.txt', 'r') as f:
    lines_mixed = f.readlines()

# Assuming the format in result_mixed.txt is:
# Basic Times
# Basic Memories
# Shared Times
# Shared Memories
fp16_basic_times = [float(x) for x in lines_mixed[0].split(':')[1].strip().split()]
fp16_basic_mems = [int(x) for x in lines_mixed[1].split(':')[1].strip().split()]
fp16_shared_times = [float(x) for x in lines_mixed[2].split(':')[1].strip().split()]
fp16_shared_mems = [int(x) for x in lines_mixed[3].split(':')[1].strip().split()]

# Time comparison plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, fp32_basic_times, label='FP32 Basic', marker='o')
plt.plot(grid_sizes, fp32_shared_times, label='FP32 Shared', marker='x')
plt.plot(grid_sizes, fp16_basic_times, label='FP16 Basic', marker='^')
plt.plot(grid_sizes, fp16_shared_times, label='FP16 Shared', marker='s')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Execution Time (s)')
plt.title('Time Comparison: FP32 CUDA vs FP16 Mixed')
plt.legend()
plt.grid()
plt.show()

# Memory comparison plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, fp32_basic_mems, label='FP32 Basic', marker='o')
plt.plot(grid_sizes, fp32_shared_mems, label='FP32 Shared', marker='x')
plt.plot(grid_sizes, fp16_basic_mems, label='FP16 Basic', marker='^')
plt.plot(grid_sizes, fp16_shared_mems, label='FP16 Shared', marker='s')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Peak GPU Memory (MiB)')
plt.title('Memory Comparison: FP32 CUDA vs FP16 Mixed')
plt.legend()
plt.grid()
plt.show()

# Speedup plot (FP16 speedup over FP32 baseline)
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, [fp32 / fp16 for fp32, fp16 in zip(fp32_basic_times, fp16_basic_times)], label='Basic Speedup', marker='x')
plt.plot(grid_sizes, [fp32 / fp16 for fp32, fp16 in zip(fp32_shared_times, fp16_shared_times)], label='Shared Speedup', marker='^')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Speedup (time_FP32 / time_FP16)')
plt.title('FP16 Mixed Speedup vs FP32 CUDA Baseline')
plt.legend()
plt.grid()
plt.show()