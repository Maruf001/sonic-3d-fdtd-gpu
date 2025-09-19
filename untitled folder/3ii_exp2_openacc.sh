# Experiments and Benchmark (Time and Memory)

%%shell
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
sizes=(64 128 256 512 1024)
times=()
mems=()
for size in "${sizes[@]}"; do
  echo "Running for grid size $size^3"
  rm -f mem.log
  (while kill -0 $$ 2> /dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> mem.log; sleep 0.1; done) &
  monitor_pid=$!
  ./fdtd_openacc $size > output_$size.txt
  kill $monitor_pid > /dev/null 2>&1
  total_time=$(grep "Total Time" output_$size.txt | awk '{print $3}')
  peak_mem=$(sort -n mem.log | tail -1)
  echo "Peak GPU Memory: $peak_mem MiB"
  echo "Total Time: $total_time s"
  times+=($total_time)
  mems+=($peak_mem)
done
echo "Grid Sizes: ${sizes[@]}" > result_openacc.txt
echo "Times: ${times[@]}" >> result_openacc.txt
echo "Memories: ${mems[@]}" >> result_openacc.txt
echo "Grid Sizes: ${sizes[@]}"
echo "Times: ${times[@]}"
echo "Memories: ${mems[@]}"