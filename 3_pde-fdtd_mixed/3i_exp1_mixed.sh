# Experiments and Benchmark (Time and Memory)

%%shell
sizes=(64 128 256 512 1024)
times_basic=()
mems_basic=()
times_shared=()
mems_shared=()

for size in "${sizes[@]}"; do
  echo "Running basic mixed for grid size $size^3"
  rm -f mem.log
  (while [ -e /proc/$$ ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> mem.log; sleep 0.1; done) &
  monitor_pid=$!
  ./fdtd_mixed $size 0 > output_basic_$size.txt
  kill $monitor_pid > /dev/null 2>&1
  total_time=$(grep "Total Time" output_basic_$size.txt | awk '{print $3}')
  peak_mem=$(sort -n mem.log | tail -1)
  echo "Peak GPU Memory: $peak_mem MiB"
  echo "Total Time: $total_time s"
  times_basic+=($total_time)
  mems_basic+=($peak_mem)

  echo "Running shared mixed for grid size $size^3"
  rm -f mem.log
  (while [ -e /proc/$$ ]; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> mem.log; sleep 0.1; done) &
  monitor_pid=$!
  ./fdtd_mixed $size 1 > output_shared_$size.txt
  kill $monitor_pid > /dev/null 2>&1
  total_time=$(grep "Total Time" output_shared_$size.txt | awk '{print $3}')
  peak_mem=$(sort -n mem.log | tail -1)
  echo "Peak GPU Memory: $peak_mem MiB"
  echo "Total Time: $total_time s"
  times_shared+=($total_time)
  mems_shared+=($peak_mem)
done

# show results summary:
echo "Grid Sizes: ${sizes[@]}"
echo "Basic Times: ${times_basic[@]}"
echo "Basic Memories: ${mems_basic[@]}"
echo "Shared Times: ${times_shared[@]}"
echo "Shared Memories: ${mems_shared[@]}"


# Save to text file
echo "Grid Sizes: ${sizes[@]}" > results.txt
echo "Basic Times: ${times_basic[@]}" >> result_mixed.txt
echo "Basic Memories: ${mems_basic[@]}" >> result_mixed.txt
echo "Shared Times: ${times_shared[@]}" >> result_mixed.txt
echo "Shared Memories: ${mems_shared[@]}" >> result_mixed.txt