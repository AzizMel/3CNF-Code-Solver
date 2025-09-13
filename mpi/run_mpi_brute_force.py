#!/usr/bin/env python3
"""
Simple script to run MPI brute force solver
Usage: mpiexec -n <num_processes> python run_mpi_brute_force.py
"""

from mpi_brute_force_solver import MPIBruteForceSolver, create_test_formulas


def main():
    """Run MPI brute force solver with test formulas"""

    # Initialize MPI solver
    mpi_solver = MPIBruteForceSolver()

    if mpi_solver.rank == 0:
        print("=" * 60)
        print("MPI BRUTE FORCE SOLVER TEST")
        print(f"Running on {mpi_solver.size} processes")
        print("=" * 60)

    # Create test formulas
    test_formulas = create_test_formulas()

    # Run the solver
    mpi_solver.run_performance_analysis(test_formulas)


if __name__ == "__main__":
    main()
