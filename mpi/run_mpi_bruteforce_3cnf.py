#!/usr/bin/env python3
"""
MPI brute-force solver for fixed 3-CNF over variables ['x1','x2','x3'].

- Enumerates assignments explicitly (brute force): a in [0..7]
- Uses your BruteForceSolver.evaluate_formula(assignment_dict)
- Cooperative early-stop: periodic allreduce to stop others when any rank finds a solution
- Solves ALL formulas from formulas_data.formulas in ONE MPI launch
- Saves per-process-count JSON under Results/MPI/

Run:
  mpirun -n 4 python mpi/run_mpi_bruteforce_3cnf.py
  # or on Slurm:
  # srun -n 4 python mpi/run_mpi_bruteforce_3cnf.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional

from mpi4py import MPI

# Ensure project root on path so Solvers.* and formulas_data are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import user's solver classes
from Solvers.brute_force_solver import BruteForceSolver

# Fixed 3-CNF variables (order matters for assignment mapping)
FIXED_VARS: List[str] = ["x1", "x2", "x3"]


def load_formulas() -> Dict[int, List[List[str]]]:
    """Load formulas dict: {id: CNF}, else fallback demo."""
    try:
        import formulas_data
        formulas = getattr(formulas_data, "formulas", {})
        if isinstance(formulas, dict) and formulas:
            return formulas
    except Exception:
        pass
    # Fallback demo (replace freely with your own)
    return {
        1: [["x1","x2","-x3"], ["-x1","-x2","x3"], ["-x1","x2","x3"]],
        2: [["-x1","-x2","-x3"], ["x1","-x2","x3"], ["x1","x2","-x3"]],
    }


def assignment_from_index(a: int, variables: List[str]) -> Dict[str, bool]:
    """
    Map assignment index a (0..7) to dict {'x1':bool, 'x2':bool, 'x3':bool}
    Convention: a's bits are (x1<<2 | x2<<1 | x3)
    """
    return {
        "x1": bool((a >> 2) & 1),
        "x2": bool((a >> 1) & 1),
        "x3": bool(a & 1),
    }


def partition_range(total: int, size: int, rank: int):
    """
    Contiguous 1D block partition of [0..total-1].
    First 'rem' ranks get (base+1) items; others get 'base'.
    """
    base = total // size
    rem = total % size
    if rank < rem:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = rank * base + rem
        end = start + base
    return start, end


def solve_formula_bruteforce_mpi(clauses: List[List[str]],
                                 variables: List[str],
                                 check_every: int = 1) -> Dict[str, object]:
    """
    True brute-force across assignments:
      - Each rank enumerates its assigned a in [start..end)
      - Builds a dict assignment for each a and calls evaluate_formula(assignment)
      - Periodically performs an allreduce (LOR) to stop early if someone found a solution
      - Final owner election (MIN) + broadcast of the found assignment dict
    Returns (on rank 0) a dict with 'solved', 'solution', 'assignments_checked', etc.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    TOTAL = 8  # 2^3
    start, end = partition_range(TOTAL, size, rank)

    solver = BruteForceSolver(formula=clauses)
    solver.variables = variables

    local_found = False
    local_solution: Optional[Dict[str, bool]] = None
    checked = 0

    # Enumerate assigned indices (brute-force evaluation)
    for idx in range(start, end):
        checked += 1
        assignment = assignment_from_index(idx, variables)
        if solver.evaluate_formula(assignment):
            local_found = True
            local_solution = assignment
            break

        # Cooperative early-stop (cheap: boolean allreduce); set check_every=1 since TOTAL=8
        if (idx - start + 1) % check_every == 0:
            if comm.allreduce(local_found, op=MPI.LOR):
                break

    # Final "someone found?" (flush) and owner election
    any_found = comm.allreduce(local_found, op=MPI.LOR)
    if any_found:
        # Owner is the lowest rank where a solution was found
        owner_candidate = rank if local_found else 10**9
        owner = comm.allreduce(owner_candidate, op=MPI.MIN)
        # Broadcast solution dict from owner
        solution = comm.bcast(local_solution if rank == owner else None, root=owner)
    else:
        owner = 10**9
        solution = None

    # Gather counters (optional)
    per_rank_checked = comm.gather(checked, root=0)

    if rank == 0:
        return {
            "solved": solution is not None,
            "solution": solution,
            "assignments_checked": sum(per_rank_checked) if per_rank_checked else checked,
            "per_rank_checked": per_rank_checked,
            "total_assignments": TOTAL,
            "processes": size
        }
    return {}


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    formulas = load_formulas()

    if rank == 0:
        print(f"[Rank 0] MPI brute-force 3-CNF over (x1,x2,x3) | processes = {size}")
        print(f"[Rank 0] Total formulas: {len(formulas)}")
        os.makedirs("Results/MPI", exist_ok=True)

    t0 = time.time()
    results = {}

    # Solve all formulas in one MPI run (to avoid launcher overheads)
    for fid, clauses in formulas.items():
        ret = solve_formula_bruteforce_mpi(clauses, FIXED_VARS, check_every=1)
        if rank == 0:
            results[str(fid)] = {
                "formula_id": fid,
                "variables": FIXED_VARS,
                "formula": clauses,
                "mpi_bruteforce_solver": ret
            }
            print(f"[Rank 0] Formula {fid}: solved={ret['solved']}, "
                  f"checked={ret['assignments_checked']}/{ret['total_assignments']}")

    if rank == 0:
        out = f"Results/MPI/mpi_bruteforce_results_{size}proc.json"
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
