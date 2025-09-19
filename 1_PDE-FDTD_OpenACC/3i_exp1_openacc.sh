# Experiments and Benchmark (Time and Memory)

%%shell
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH

sizes=(64 128 256)
times=()
mems=()

for size in "${sizes[@]}"; do
  echo "Running for grid size $size^3"
  nvidia-smi  # Before run
  ./fdtd_openacc $size > output_$size.txt
  peak_mem=$(nvidia-smi | grep MiB | awk '{print $9}' | head -1 | sed 's/MiB//')
  nvidia-smi  # After run
  total_time=$(grep "Total Time" output_$size.txt | awk '{print $3}')
  echo "Peak GPU Memory: $peak_mem MiB"
  echo "Total Time: $total_time s"
  times+=($total_time)
  mems+=($peak_mem)
done

# Output results for Python plotting
echo "Grid Sizes: ${sizes[@]}"
echo "Times: ${times[@]}"
echo "Memories: ${mems[@]}"