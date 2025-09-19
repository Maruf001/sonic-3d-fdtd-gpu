# Benchmark analysis

import matplotlib.pyplot as plt


# grid_sizes = [64, 128, 256, 512, 1024]

# Read grid sizes from OpenACC file (consistent across all)
with open('result_openacc.txt', 'r') as f:
    lines_openacc = f.readlines()

grid_sizes = [int(x) for x in lines_openacc[0].split(':')[1].strip().split()]

# Read from result_mixed.txt
with open('result_mixed.txt', 'r') as f:
    lines = f.readlines()

# Parse the lines - Adjusted to read time and memory based on the fixed format
basic_times = [float(x) for x in lines[0].split(':')[1].strip().split()]
basic_memories = [int(x) for x in lines[1].split(':')[1].strip().split()]
shared_times = [float(x) for x in lines[2].split(':')[1].strip().split()]
shared_memories = [int(x) for x in lines[3].split(':')[1].strip().split()]

# Time plot
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Grid Size (n^3)')
ax1.set_ylabel('Execution Time (s)', color='tab:blue')
ax1.plot(grid_sizes, basic_times, color='tab:blue', marker='o', label='Basic')
ax1.plot(grid_sizes, shared_times, color='tab:green', marker='x', label='Shared')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

plt.title('FP16 Performance: Time vs Grid Size')
plt.grid()
plt.show()

# Memory plot
fig, ax2 = plt.subplots(figsize=(10, 5))

ax2.set_xlabel('Grid Size (n^3)')
ax2.set_ylabel('Peak GPU Memory (MiB)', color='tab:orange')
ax2.plot(grid_sizes, basic_memories, color='tab:orange', marker='o', label='Basic')
ax2.plot(grid_sizes, shared_memories, color='tab:purple', marker='x', label='Shared')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper left')

plt.title('FP16 Performance: Memory vs Grid Size')
plt.grid()
plt.show()