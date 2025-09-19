# Benchmark analysis

import matplotlib.pyplot as plt

# Read data from OpenACC file
with open('result_openacc.txt', 'r') as f:
    lines_openacc = f.readlines()

grid_sizes = [int(x) for x in lines_openacc[0].split(':')[1].strip().split()]
openacc_times = [float(x) for x in lines_openacc[1].split(':')[1].strip().split()]
openacc_mems = [int(x) for x in lines_openacc[2].split(':')[1].strip().split()]

# Read data from CUDA file
with open('result_cuda.txt', 'r') as f:
    lines_cuda = f.readlines()

# Read time and memory based on line indices from result_cuda.txt 
cuda_basic_times = [float(x) for x in lines_cuda[1].split(':')[1].strip().split()]
cuda_basic_mems = [int(x) for x in lines_cuda[2].split(':')[1].strip().split()]
cuda_shared_times = [float(x) for x in lines_cuda[3].split(':')[1].strip().split()]
cuda_shared_mems = [int(x) for x in lines_cuda[4].split(':')[1].strip().split()]


# Time plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, openacc_times, label='OpenACC', marker='o')
plt.plot(grid_sizes, cuda_basic_times, label='CUDA Basic', marker='x')
plt.plot(grid_sizes, cuda_shared_times, label='CUDA Shared', marker='^')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Execution Time (s)')
plt.title('Time Comparison: OpenACC vs CUDA Variants')
plt.legend()
plt.grid()
plt.show()

# Memory plot
plt.figure(figsize=(10, 5))
plt.plot(grid_sizes, openacc_mems, label='OpenACC', marker='o')
plt.plot(grid_sizes, cuda_basic_mems, label='CUDA Basic', marker='x')
plt.plot(grid_sizes, cuda_shared_mems, label='CUDA Shared', marker='^')
plt.xlabel('Grid Size (n^3)')
plt.ylabel('Peak GPU Memory (MiB)')
plt.title('Memory Comparison: OpenACC vs CUDA Variants')
plt.legend()
plt.grid()
plt.show()