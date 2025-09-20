#!/usr/bin/env python3
"""
FAST MPI brute-force solver for 3-CNF with fixed variables ['x1','x2','x3'].

Why this is fast:
- True brute-force: enumerate all 8 assignments explicitly.
- NO collectives inside the inner loop. Per formula we do ONLY:
    1) MPI.Allreduce(MIN) to elect the owner rank (if any rank found a solution)
    2) MPI.Bcast to share the solution dict
- Single run over all formulas. No subcommunicators, no virtual-np sweeps, no barriers.

Run (Open MPI):
  mpiexec --bind-to none -n 80 python mpi/run_mpi_bruteforce_3cnf_fast.py

Run (Slurm):
  srun -N1 -n80 python mpi/run_mpi_bruteforce_3cnf_fast.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional

from mpi4py import MPI

# Ensure project root on path so we can import formulas_data (and optionally your solver)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Fixed variables (order defines bit mapping)
FIXED_VARS: List[str] = ["x1", "x2", "x3"]
TOTAL_ASSIGNMENTS = 8  # 2^3

# --- Try to import your BruteForceSolver; if unavailable, use internal evaluator ---
try:
    from Solvers.brute_force_solver import BruteForceSolver  # optional
except Exception:
    BruteForceSolver = None

# --- Load formulas (required) ---
def load_formulas() -> Dict[int, List[List[str]]]:
    import formulas_data
    formulas = getattr(formulas_data, "formulas", {})
    if not isinstance(formulas, dict) or not formulas:
        raise RuntimeError("formulas_data.formulas is missing or empty")
    return formulas

# --- Helpers ---
def assignment_from_index(a: int) -> Dict[str, bool]:
    # bits: a = (x1<<2) | (x2<<1) | (x3<<0)
    return {
        "x1": bool((a >> 2) & 1),
        "x2": bool((a >> 1) & 1),
        "x3": bool(a & 1),
    }

def partition_range(total: int, size: int, rank: int):
    base = total // size
    rem  = total %  size
    if rank < rem:
        start = rank * (base + 1)
        end   = start + base + 1
    else:
        start = rank * base + rem
        end   = start + base
    return start, end

# Internal CNF evaluator (used if your Solvers module isn't available)
def evaluate_formula_python(clauses: List[List[str]], assignment: Dict[str,bool]) -> bool:
    def eval_clause(clause):
        for lit in clause:
            if lit.startswith("-"):
                v = lit[1:]
                if not assignment.get(v, False):
                    return True
            else:
                if assignment.get(lit, False):
                    return True
        return False
    return all(eval_clause(c) for c in clauses)

def solve_formula_bruteforce(comm: MPI.Comm,
                             clauses: List[List[str]]) -> Dict[str, object]:
    """
    True brute-force (enumeration) with ZERO collectives inside the inner loop.
    Per formula collectives: 1x Allreduce(MIN) + 1x Bcast.
    Returns a dict on rank 0; {} on other ranks.
    """
    rank = comm.Get_rank()
    size = comm.Get_Size() if hasattr(comm, "Get_Size") else comm.Get_size()

    start, end = partition_range(TOTAL_ASSIGNMENTS, size, rank)

    # Choose evaluator
    solver = BruteForceSolver(formula=clauses) if BruteForceSolver else None
    if solver:
        solver.variables = FIXED_VARS

    local_found = False
    local_solution: Optional[Dict[str, bool]] = None

    # --- Inner brute-force, NO collectives here ---
    for idx in range(start, end):
        assignment = assignment_from_index(idx)
        ok = solver.evaluate_formula(assignment) if solver else evaluate_formula_python(clauses, assignment)
        if ok:
            local_found = True
            local_solution = assignment
            break

    # --- One collective to elect owner rank ---
    owner_candidate = rank if local_found else 10**9
    owner = comm.allreduce(owner_candidate, op=MPI.MIN)

    # --- Broadcast solution from owner if any ---
    if owner < 10**9:
        solution = comm.bcast(local_solution if rank == owner else None, root=owner)
    else:
        solution = None

    # If you don't care about per-rank counters, skip gather to avoid extra collective.
    # We'll keep it minimal: no gather.
    if rank == 0:
        return {
            "solved": solution is not None,
            "solution": solution,
            "total_assignments": TOTAL_ASSIGNMENTS,
            "processes": size
        }
    return {}

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    formulas = load_formulas()

    if rank == 0:
        print(f"[Rank 0] MPI brute-force 3-CNF | processes = {size}")
        print(f"[Rank 0] Total formulas: {len(formulas)}")
        os.makedirs("Results/MPI", exist_ok=True)

    t0 = time.time()
    results = {}

    # Solve all formulas (one run, minimal comms)
    for fid, clauses in formulas.items():
        ret = solve_formula_bruteforce(comm, clauses)
        if rank == 0:
            results[str(fid)] = {
                "formula_id": fid,
                "variables": FIXED_VARS,
                "formula": clauses,
                "mpi_bruteforce": ret
            }

    if rank == 0:
        out = f"Results/MPI/mpi_bruteforce_fast_{size}proc.json"
        with open(out, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "processes": size,
                "runtime_seconds": time.time() - t0,
                "results": results
            }, f, indent=2)
        print(f"[Rank 0] Results saved to {out}")

if __name__ == "__main__":
    main()
