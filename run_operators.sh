#!/usr/bin/env bash
set -euo pipefail

# Ops to run
ops=(
  # bf16xint16_gemm
  # # fp8_gemm no support
  # gather_gemv
  # gdpa
  # gemm
  # grouped_gemm
  # # int4_gemm has error
  # # jagged_mean can not finish
  # # jagged_softmax can not finish
  # # jagged_sum no support/can not finish
  # # mixed_gemm
  # # sum
  # # template_attention
  # welford
)

mkdir -p ./run

# Prefer GNU time if available
if command -v /usr/bin/time >/dev/null 2>&1; then
  TIME_CMD=/usr/bin/time
elif command -v gtime >/dev/null 2>&1; then
  TIME_CMD=$(command -v gtime)   # macOS with coreutils
else
  TIME_CMD=time                  # shell builtin
fi

# Format: <command>   <user>s user <sys>s system <cpu>% cpu <real> total
TIME_FMT="%C   %Us user %Ss system %P cpu %e total"

# Number of repeats for each op
REPEAT=5

for op in "${ops[@]}"; do
  out="./run/${op}.csv"
  cmd=(env TRITON_PRINT_AUTOTUNING=1 python run.py --op "$op")

  # Header
  {
    echo "# $(date -u +"%Y-%m-%dT%H:%M:%SZ")  ${cmd[*]}"
    echo "# Working dir: $(pwd)"
    echo "# Runs: $REPEAT"
    echo
  } > "$out"

  for i in $(seq 1 $REPEAT); do
    echo "# Run $i" >>"$out"
    if "$TIME_CMD" --version >/dev/null 2>&1; then
      # GNU time
      if [[ "${LIVE:-0}" == "1" ]]; then
        ( "$TIME_CMD" -f "$TIME_FMT" -- "${cmd[@]}" ) 2>&1 | tee -a "$out"
      else
        ( "$TIME_CMD" -f "$TIME_FMT" -- "${cmd[@]}" ) >>"$out" 2>&1
      fi
    else
      # Builtin time
      if [[ "${LIVE:-0}" == "1" ]]; then
        ( time "${cmd[@]}" ) 2>&1 | tee -a "$out"
      else
        ( time "${cmd[@]}" ) >>"$out" 2>&1
      fi
    fi
    echo >>"$out"
  done
done
