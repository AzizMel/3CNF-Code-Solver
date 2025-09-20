#!/usr/bin/env python3
"""
MPI brute force solver runner script
Usage: mpiexec -n <num_processes> python run_mpi_brute_force.py
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the Python path to find the Solvers module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from mpi_brute_force_solver import MPIBruteForceSolver
import formulas_data


def main():
    """Run MPI brute force solver with test formulas"""

    # Initialize MPI solver
    mpi_solver = MPIBruteForceSolver()

    if mpi_solver.rank == 0:
        print("=" * 60)
        print("MPI BRUTE FORCE SOLVER TEST")
        print(f"Running on {mpi_solver.size} processes")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    # Run the solver
    mpi_solver.run_performance_analysis()

    if mpi_solver.rank == 0:
        print("\n" + "=" * 60)
        print("MPI SOLVER COMPLETED")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
