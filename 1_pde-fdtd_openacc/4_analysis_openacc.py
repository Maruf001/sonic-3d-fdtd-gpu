# Benchmark analysis

import matplotlib.pyplot as plt

# Read data from the exported file
with open('result_openacc.txt', 'r') as f:
    lines = f.readlines()

grid_sizes = [int(x) for x in lines[0].split(':')[1].strip().split()]
times = [float(x) for x in lines[1].split(':')[1].strip().split()]
memories = [int(x) for x in lines[2].split(':')[1].strip().split()]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Grid Size (n^3)')
ax1.set_ylabel('Execution Time (s)', color='tab:blue')
ax1.plot(grid_sizes, times, color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Peak GPU Memory (MiB)', color='tab:orange')
ax2.plot(grid_sizes, memories, color='tab:orange', marker='x')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('OpenACC Performance: Time and Memory vs. Grid Size')
plt.show()