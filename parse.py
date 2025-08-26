#!/usr/bin/env python3
"""
Parse Triton autotuning logs under ./run/ produced by the run script.

What it extracts:
1) For each "group" (a sequence of "Autotuning kernel ... with config ..." lines
   followed by a "Triton autotuning for function ..." footer), it records:
   - file (op)
   - run index (if present via "# Run N" markers; else 1)
   - group index within the run
   - function name
   - number of config lines in the group
   - "best config selected" string
   - group tuning time in seconds (from "finished after Xs,")

2) For each run, it extracts the timing line from the shell `time` output like:
   TRITON_PRINT_AUTOTUNING=1 python run.py --op ...  6.71s  user 5.06s system 173% cpu 6.799 total

   and records user_s, sys_s, cpu_pct, real_s.

3) It also computes the average of the *first* time value ("user" seconds) across runs
   of the same file, matching the user's example.

Outputs (written next to this script's working directory by default):
- autotune_groups.csv : per-group data
- op_times.csv        : per-file, per-run timing rows (+ per-file averages as final rows)

Usage:
    python parse_autotune.py [run_dir]
If run_dir is omitted, defaults to ./run
"""

from pathlib import Path
import re
import csv
import sys
from typing import List, Dict, Any

CONFIG_LINE_RE = re.compile(
    r'^Autotuning kernel\s+(?P<kernel>\S+)\s+with config\s+(?P<cfg>.+)$'
)

GROUP_START_RE = re.compile(
    r'^Triton autotuning for function\s+(?P<func>\S+),\s*$'
)

FINISHED_RE = re.compile(
    r'^finished after\s+(?P<secs>[0-9.]+)s,?\s*$'
)

BEST_CONFIG_RE = re.compile(
    r'^best config selected:\s+(?P<cfg>.+?);?\s*$'
)

# Example:
# TRITON_PRINT_AUTOTUNING=1 python run.py --op bf16xint16_gemm   6.71s  user 5.06s system 173% cpu 6.799 total
TIME_LINE_RE = re.compile(
    r'^env TRITON_PRINT_AUTOTUNING=1\s+python\s+run\.py\s+--op\s+\S+\s+'
    r'(?P<user>[0-9.]+)s\s+user\s+'
    r'(?P<sys>[0-9.]+)s\s+system\s+'
    r'(?P<cpu>[0-9.]+)%\s+cpu\s+'
    r'(?P<real>[0-9.]+)\s+total\s*$'
)

RUN_MARK_RE = re.compile(r'^#\s*Run\s+(?P<idx>\d+)\s*$')

def parse_file(path: Path) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    times: List[Dict[str, Any]] = []

    # State for parsing groups
    run_idx = 1
    group_idx = 0
    in_group = False
    pending_configs = 0
    current_func = None
    current_group: Dict[str, Any] = {}

    with path.open('r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\n')

            # Run markers (from the bash logger); helps separate repeats
            m = RUN_MARK_RE.match(line)
            if m:
                run_idx = int(m.group('idx'))
                # Reset group index for each run
                group_idx = 0
                continue

            # Count config lines
            m = CONFIG_LINE_RE.match(line)
            if m:
                # Start of a group or continuation
                if not in_group:
                    in_group = True
                    group_idx += 1
                    pending_configs = 0
                    current_func = None
                    current_group = {
                        'file': path.name,
                        'run': run_idx,
                        'group': group_idx,
                        'function': None,
                        'num_configs': 0,
                        'best_config': None,
                        'group_time_s': None,
                    }
                pending_configs += 1
                continue

            # Group footer start
            m = GROUP_START_RE.match(line)
            if m and in_group:
                current_func = m.group('func')
                current_group['function'] = current_func
                continue

            # Finished time
            m = FINISHED_RE.match(line)
            if m and in_group:
                current_group['group_time_s'] = float(m.group('secs'))
                continue

            # Best config
            m = BEST_CONFIG_RE.match(line)
            if m and in_group:
                current_group['best_config'] = m.group('cfg')
                # Group closes once we've seen best config
                current_group['num_configs'] = pending_configs
                groups.append(current_group)
                in_group = False
                pending_configs = 0
                current_group = {}
                current_func = None
                continue

            # Parse time line
            m = TIME_LINE_RE.match(line)
            if m:
                times.append({
                    'file': path.name,
                    'run': run_idx,
                    'user_s': float(m.group('user')),
                    'sys_s': float(m.group('sys')),
                    'cpu_pct': float(m.group('cpu')),
                    'real_s': float(m.group('real')),
                })
                continue

    # Edge case: if file ended mid-group without best config, finalize with what we have
    if in_group:
        current_group['num_configs'] = pending_configs
        groups.append(current_group)

    return {'groups': groups, 'times': times}

def main(run_dir: str = "./run"):
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[ERROR] Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    all_groups: List[Dict[str, Any]] = []
    all_times: List[Dict[str, Any]] = []

    files = sorted(run_path.glob("*.csv"))
    if not files:
        print(f"[WARN] No .csv files in {run_dir}")
    for p in files:
        parsed = parse_file(p)
        all_groups.extend(parsed['groups'])
        all_times.extend(parsed['times'])

    # Write group summary
    with open("autotune_groups.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'file', 'run', 'group', 'function', 'num_configs', 'group_time_s', 'best_config'
        ])
        writer.writeheader()
        for row in all_groups:
            writer.writerow(row)

    # Write per-run timing and per-file averages
    with open("op_times.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'file', 'run', 'user_s', 'sys_s', 'cpu_pct', 'real_s'
        ])
        writer.writeheader()
        for row in all_times:
            writer.writerow(row)

        # Compute averages by file for the "user_s" (per user's request),
        # but also include other averages as convenience.
        from collections import defaultdict
        by_file = defaultdict(list)
        for row in all_times:
            by_file[row['file']].append(row)

        # Add an empty line separator in CSV via comment-ish row
        # (CSV doesn't support comments; we add identifiable "AVERAGE" rows)
        for fname, rows in sorted(by_file.items()):
            n = len(rows)
            avg_user = sum(r['user_s'] for r in rows) / n if n else 0.0
            avg_sys  = sum(r['sys_s']  for r in rows) / n if n else 0.0
            avg_cpu  = sum(r['cpu_pct'] for r in rows) / n if n else 0.0
            avg_real = sum(r['real_s'] for r in rows) / n if n else 0.0
            writer.writerow({
                'file': f"{fname} (AVERAGE over {n} runs)",
                'run':  'avg',
                'user_s': round(avg_user, 6),
                'sys_s':  round(avg_sys, 6),
                'cpu_pct': round(avg_cpu, 3),
                'real_s': round(avg_real, 6),
            })

    # Print a human-readable summary
    print("Wrote: autotune_groups.csv  (per-group best config, count, and group time)")
    print("Wrote: op_times.csv        (per-run time and per-file averages)")
    if all_times:
        # Show a quick average table in stdout
        from collections import defaultdict
        by_file = defaultdict(list)
        for row in all_times:
            by_file[row['file']].append(row['user_s'])
        print("\nAverage 'user' seconds by file:")
        for fname, vals in sorted(by_file.items()):
            avg_user = sum(vals)/len(vals)
            print(f"  {fname}: {avg_user:.3f}s over {len(vals)} runs")
    else:
        print("No timing lines found; check your log format.")

if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "./run"
    main(run_dir)
