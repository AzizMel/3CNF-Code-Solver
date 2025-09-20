#!/usr/bin/env python3
"""
Scaling runner for the brute-force 3-CNF MPI solver (x1,x2,x3).
Launches multiple MPI runs for selected process counts, collects JSON outputs,
and writes a combined report.

Usage examples:
  python run_mpi_scaling.py
  python run_mpi_scaling.py --procs 1,2,4,8,16 --timeout 180 --mpiexec-cmd mpirun
  python run_mpi_scaling.py --min 1 --max 52 --timeout 300
  # On Slurm:
  python run_mpi_scaling.py --mpiexec-cmd srun --procs 1,2,4,8,16
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.abspath(__file__))
MPI_SCRIPT = os.path.join(ROOT, "mpi", "run_mpi_bruteforce_3cnf_singlelaunch_scaling.py")

def parse_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=int, default=1, help="Min process count (used if --procs not given)")
    p.add_argument("--max", type=int, default=56, help="Max process count (used if --procs not given)")
    p.add_argument("--procs", type=str, default="1,2,4,8,16,32,56", help="Comma-separated list, e.g., 1,2,4,8,16")
    p.add_argument("--timeout", type=int, default=300, help="Timeout per run (seconds)")
    p.add_argument("--mpiexec-cmd", type=str, default="mpiexec", help="mpiexec or srun")
    return p.parse_args()

def run_once(np_count: int, mpiexec_cmd: str, timeout_s: int) -> Dict[str, Any]:
    py = sys.executable  # venv python
    cmd = [mpiexec_cmd, "--bind-to", "none", "-n", str(np_count), py, MPI_SCRIPT]

    print("\n" + "=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print(f"Timeout: {timeout_s} s")
    print("=" * 60)

    start = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=ROOT)
        duration = time.time() - start
        results_path = os.path.join(ROOT, "Results", "MPI", f"mpi_bruteforce_results_{np_count}proc.json")

        payload = None
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                payload = json.load(f)

        return {
            "processes": np_count,
            "returncode": res.returncode,
            "duration_sec": duration,
            "stdout_tail": "\n".join(res.stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(res.stderr.splitlines()[-20:]),
            "results_file": results_path if os.path.exists(results_path) else None,
            "results": payload
        }
    except subprocess.TimeoutExpired:
        return {
            "processes": np_count,
            "returncode": -1,
            "duration_sec": timeout_s,
            "error": f"timeout after {timeout_s} s"
        }

def main():
    args = parse_cli()

    if not os.path.exists(MPI_SCRIPT):
        print(f"ERROR: missing {MPI_SCRIPT}")
        sys.exit(1)

    if args.procs:
        procs = [int(x) for x in args.procs.split(",") if x.strip()]
    else:
        procs = list(range(args.min, args.max + 1))

    runs: List[Dict[str, Any]] = []
    os.makedirs(os.path.join(ROOT, "Results", "MPI"), exist_ok=True)

    for np_count in procs:
        runs.append(run_once(np_count, args.mpiexec_cmd, args.timeout))
        time.sleep(0.3)

    report = os.path.join(ROOT, "Results", "MPI", f"scaling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "runs": runs}, f, indent=2)

    successes = sum(1 for r in runs if r.get("returncode") == 0)
    print("\nAll runs complete.")
    print(f"Successful runs: {successes}/{len(runs)}")
    print(f"Combined report: {report}")

if __name__ == "__main__":
    main()
