# Experiments and Benchmark (Time and Memory)

%%bash
set -euo pipefail

OUT="result_cuda_python.txt"   # CSV compatible with your earlier plotting code
echo "grid_size,time_basic_s,peak_mem_basic_mib,time_shared_s,peak_mem_shared_mib" > "$OUT"

sizes=(64 128 256 512 1024)

run_one() {
  local size="$1"
  local use_shared="$2"  # 0=basic, 1=shared
  local tag out log peak total

  tag="out_${size}_${use_shared}.txt"
  log="mem_${size}_${use_shared}.log"
  : > "$log"

  # Launch the benchmark *in the background* so we can sample *its* PID
  ./fdtd_cuda "$size" "$use_shared" > "$tag" &
  pid=$!

  # Sample only this process's GPU memory (not the whole board)
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
      | awk -v p="$pid" '$1==p {print $2}' >> "$log"
    sleep 0.1
  done
  wait "$pid"

  # Extract time from program output
  total=$(awk '/Total Time/{print $3; exit}' "$tag")

  # Peak process GPU mem (MiB); handle very fast runs with no samples
  if [[ -s "$log" ]]; then
    peak=$(sort -n "$log" | tail -1)
  else
    peak=0
  fi

  # Echo two fields so caller can `read`
  echo "$total $peak"
}

for size in "${sizes[@]}"; do
  echo "Running BASIC (size=${size}^3)"
  read tbasic pbasic < <(run_one "$size" 0)

  echo "Running SHARED (size=${size}^3)"
  read tshared pshared < <(run_one "$size" 1)

  printf "%d,%s,%s,%s,%s\n" "$size" "$tbasic" "$pbasic" "$tshared" "$pshared" >> "$OUT"
done

echo "Saved $OUT"
column -t -s, "$OUT"
