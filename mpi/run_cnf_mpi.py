#!/usr/bin/env python3
"""
Ultra-fast MPI solver for 3-variable CNF (x1, x2, x3) with any number of clauses.

Key ideas:
- 8 assignments total -> represent truth tables as 8-bit masks.
- Clause mask = OR of literal masks. Formula mask = AND of clause masks.
- Each rank has a "range mask" (its subset of assignments). Match = (formula_mask & range_mask) != 0.
- Zero per-assignment loops; only a handful of bit ops per formula.
- Exactly two collectives per formula: MIN-reduce (owner election) + Bcast (solution index).

Run:
  mpirun -n 4 python mpi/run_cnf_mpi.py
  # or on Slurm: srun -n 4 python mpi/run_cnf_mpi.py

Results saved to: Results/MPI/mpi_results_<proc>proc.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

from mpi4py import MPI

# Ensure project root on path in case you import external modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------------------------
# Literal masks for 3 variables (assignment index a in [0..7], bits: x1=bit2, x2=bit1, x3=bit0)
# A bit (1<<a) is 1 if the literal is True under assignment 'a'.
# Precomputed:
#   x1 True  at a in {4,5,6,7} => 0b11110000 = 240
#   x1 False at a in {0,1,2,3} => 0b00001111 = 15
#   x2 True  at a in {2,3,6,7} => 0b11001100 = 204
#   x2 False at a in {0,1,4,5} => 0b00110011 = 51
#   x3 True  at a in {1,3,5,7} => 0b10101010 = 170
#   x3 False at a in {0,2,4,6} => 0b01010101 = 85
# --------------------------------------------------------------------------------------
LITERAL_MASK: Dict[str, int] = {
    "x1":  0b11110000, "-x1": 0b00001111,
    "x2":  0b11001100, "-x2": 0b00110011,
    "x3":  0b10101010, "-x3": 0b01010101,
}

FIXED_VARS: List[str] = ["x1", "x2", "x3"]  # for reporting only


def load_formulas() -> Dict[int, List[List[str]]]:
    """
    Load formulas from formulas_data.formulas if present, otherwise use a demo set.
    Each formula is: List[clause], each clause is: List[literal], literal in {"x1","-x1","x2","-x2","x3","-x3"}.
    """
    try:
        import formulas_data
        formulas = getattr(formulas_data, "formulas", {})
        if isinstance(formulas, dict) and formulas:
            return formulas
    except Exception:
        pass

    # Fallback demo including your multi-clause examples
    return {
        46: [["x1","x2","-x3"], ["-x1","-x2","x3"], ["-x1","x2","x3"]],
        47: [["-x1","-x2","-x3"], ["x1","-x2","x3"], ["x1","x2","-x3"]],
        48: [["x1","-x2","-x3"], ["-x1","x2","-x3"], ["-x1","-x2","x3"]],
        49: [["-x1","x2","x3"], ["x1","-x2","x3"], ["x1","x2","-x3"]],
        50: [
            ["x1","x2","x3"], ["x1","x2","-x3"], ["x1","-x2","x3"], ["x1","-x2","-x3"],
            ["-x1","x2","x3"], ["-x1","x2","-x3"], ["-x1","-x2","x3"], ["-x1","-x2","-x3"]
        ],
        51: [
            ["x1","x2","x3"], ["x1","x2","-x3"], ["x1","-x2","x3"],
            ["-x1","x2","x3"], ["-x1","-x2","x3"], ["-x1","-x2","-x3"],
            ["x1","-x2","-x3"], ["-x1","x2","-x3"]
        ],
        52: [
            ["x1","x2","x3"], ["x1","x2","-x3"], ["-x1","-x2","x3"],
            ["-x1","-x2","-x3"], ["x1","-x2","x3"], ["-x1","x2","x3"],
            ["x1","-x2","-x3"], ["-x1","x2","-x3"]
        ],
        53: [
            ["x1","x2","x3"], ["x1","-x2","x3"], ["-x1","x2","x3"],
            ["-x1","-x2","x3"], ["x1","x2","-x3"], ["x1","-x2","-x3"],
            ["-x1","x2","-x3"], ["-x1","-x2","-x3"]
        ],
        54: [
            ["x1","x2","x3"], ["x1","x2","-x3"], ["x1","-x2","x3"], ["x1","-x2","-x3"],
            ["-x1","x2","x3"], ["-x1","x2","-x3"], ["-x1","-x2","x3"], ["-x1","-x2","-x3"],
        ],
    }


def clause_mask(clause: List[str]) -> int:
    """Bitmask (8 bits) of assignments that satisfy the clause."""
    m = 0
    for lit in clause:
        try:
            m |= LITERAL_MASK[lit]
        except KeyError:
            raise ValueError(f"Unknown literal: {lit}")
    return m & 0xFF


def formula_mask(clauses: List[List[str]]) -> int:
    """Bitmask of assignments that satisfy the whole formula (AND of clause masks)."""
    m = 0xFF  # all 8 assignments initially possible
    for cl in clauses:
        m &= clause_mask(cl)
        if m == 0:
            break  # early exit if already unsatisfiable
    return m


def first_set_bit_index(mask: int) -> int:
    """Return smallest assignment index a (0..7) such that bit a is 1, or -1 if none."""
    if mask == 0:
        return -1
    lsb = mask & -mask
    return (lsb.bit_length() - 1)  # index of least significant 1-bit


def my_range_mask(rank: int, size: int) -> int:
    """
    Build the 8-bit mask representing this rank's contiguous assignment subrange.
    Partition of 8 among 'size' ranks: first 'rem' ranks get one extra.
    """
    TOTAL = 8
    base = TOTAL // size
    rem  = TOTAL %  size
    if rank < rem:
        start = rank * (base + 1)
        end   = start + base + 1
    else:
        start = rank * base + rem
        end   = start + base

    m = 0
    for a in range(start, end):
        m |= (1 << a)
    return m & 0xFF


def assignment_to_dict(a: int) -> Dict[str, bool]:
    """Map assignment index (0..7) to {'x1':bool, 'x2':bool, 'x3':bool}."""
    return {
        "x1": bool((a >> 2) & 1),
        "x2": bool((a >> 1) & 1),
        "x3": bool(a & 1),
    }


def solve_formula_mpi(clauses: List[List[str]]) -> Dict[str, object]:
    """
    Bit-mask solution with exactly two collectives per formula:
      - MIN owner election
      - Bcast solution index
    Rank 0 returns a result dict; other ranks return {}.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Precompute masks
    fmask = formula_mask(clauses)          # satisfying assignments (global)
    rmask = my_range_mask(rank, size)      # assignments owned by this rank
    local_mask = fmask & rmask             # satisfying assignments owned locally

    if local_mask:
        local_owner = rank
        local_sol_idx = first_set_bit_index(local_mask)  # choose the smallest index we own
    else:
        local_owner = 10**9
        local_sol_idx = -1

    # 1) Owner election: smallest rank that has a solution bit
    owner = comm.allreduce(local_owner, op=MPI.MIN)

    # 2) Broadcast the chosen assignment index from the owner (or -1 if none globally)
    if owner < 10**9:
        sol_idx = comm.bcast(local_sol_idx if rank == owner else None, root=owner)
    else:
        sol_idx = -1

    # Reporting: per-rank "checked" is the size of each partition (not really used now)
    # but we keep a consistent metric: number of assignment indices considered by this rank
    # (informational only; compute is O(#clauses), not O(#assignments)).
    TOTAL = 8
    base = TOTAL // size
    rem  = TOTAL %  size
    if rank < rem:
        local_count = base + 1
    else:
        local_count = base
    per_rank_counts = comm.gather(local_count, root=0)

    if rank == 0:
        solved = (sol_idx >= 0)
        return {
            "solved": solved,
            "solution": assignment_to_dict(sol_idx) if solved else None,
            "assignments_total": TOTAL,
            "per_rank_partition_sizes": per_rank_counts,
            "processes": size
        }
    return {}


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    formulas = load_formulas()

    if rank == 0:
        print(f"[Rank 0] 3-variable CNF over (x1,x2,x3) | MPI processes = {size}")
        print(f"[Rank 0] Total formulas: {len(formulas)}")
        os.makedirs("Results/MPI", exist_ok=True)

    t0 = time.time()
    results = {}

    # Solve all formulas in one MPI run
    for fid, clauses in formulas.items():
        ret = solve_formula_mpi(clauses)
        if rank == 0:
            results[str(fid)] = {
                "formula_id": fid,
                "variables": FIXED_VARS,
                "formula": clauses,
                "mpi_bitmask_solver": ret
            }
            print(f"[Rank 0] Formula {fid}: solved={ret['solved']}")

    if rank == 0:
        out = f"Results/MPI/mpi_results_{size}proc.json"
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
