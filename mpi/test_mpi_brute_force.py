#!/usr/bin/env python3
"""
Test script for MPI brute force solver (works without MPI for basic testing)
"""

import sys
import time
import itertools
from typing import List, Dict, Optional, Any

# Import the brute force solver
from Solvers.brute_force_solver import BruteForceSolver


def test_brute_force_solver():
    """Test the basic brute force solver functionality"""

    print("Testing Brute Force Solver...")
    print("=" * 50)

    # Test formulas
    test_cases = [
        {
            "name": "Simple solvable",
            "formula": [["x1", "x2"], ["-x1", "-x2"]],
            "variables": ["x1", "x2"],
            "expected_solvable": True,
        },
        {
            "name": "Simple unsolvable",
            "formula": [["x1", "x2"], ["-x1", "-x2"], ["x1", "-x2"], ["-x1", "x2"]],
            "variables": ["x1", "x2"],
            "expected_solvable": False,
        },
        {
            "name": "3-variable solvable",
            "formula": [["x1", "x2", "x3"], ["-x1", "-x2", "-x3"]],
            "variables": ["x1", "x2", "x3"],
            "expected_solvable": True,
        },
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Formula: {test_case['formula']}")
        print(f"Variables: {test_case['variables']}")

        # Create solver
        solver = BruteForceSolver(formula=test_case["formula"])
        solver.variables = test_case["variables"]

        # Solve
        start_time = time.time()
        solution = solver.solve()
        end_time = time.time()

        execution_time = end_time - start_time
        solution_found = solution is not None

        print(f"Result: {'SOLVED' if solution_found else 'UNSOLVABLE'}")
        print(f"Time: {execution_time:.6f}s")
        if solution:
            print(f"Solution: {solution}")

        # Check if result matches expectation
        expected = test_case["expected_solvable"]
        if solution_found == expected:
            print("✓ PASS")
        else:
            print(f"✗ FAIL - Expected {'solvable' if expected else 'unsolvable'}")

    print("\n" + "=" * 50)
    print("Basic brute force solver test complete!")


def test_parallel_simulation():
    """Simulate parallel processing without MPI"""

    print("\nSimulating Parallel Brute Force...")
    print("=" * 50)

    # Test with a larger formula
    formula = [["x1", "x2", "x3", "x4"], ["-x1", "-x2", "-x3", "-x4"]]
    variables = ["x1", "x2", "x3", "x4"]

    print(f"Formula: {formula}")
    print(f"Variables: {variables}")
    print(f"Total assignments: {2**len(variables)}")

    # Simulate different numbers of processes
    for num_processes in [1, 2, 4]:
        print(f"\nSimulating {num_processes} processes:")

        # Calculate work distribution
        total_assignments = 2 ** len(variables)
        assignments_per_process = total_assignments // num_processes
        remainder = total_assignments % num_processes

        print(f"  Assignments per process: ~{assignments_per_process}")

        # Simulate parallel execution
        start_time = time.time()

        # Create solver
        solver = BruteForceSolver(formula=formula)
        solver.variables = variables

        # Generate all assignments
        all_assignments = list(itertools.product([False, True], repeat=len(variables)))

        solution = None
        assignments_checked = 0

        # Simulate work distribution
        for process_id in range(num_processes):
            start_idx = process_id * assignments_per_process
            if process_id < remainder:
                start_idx += process_id
            else:
                start_idx += remainder

            end_idx = start_idx + assignments_per_process

            # Search in this process's range
            for i in range(start_idx, min(end_idx, len(all_assignments))):
                assignments_checked += 1
                assignment = dict(zip(variables, all_assignments[i]))

                if solver.evaluate_formula(assignment):
                    solution = assignment
                    break

            # If solution found, break early
            if solution:
                break

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"  Result: {'SOLVED' if solution else 'UNSOLVABLE'}")
        print(f"  Time: {execution_time:.6f}s")
        print(f"  Assignments checked: {assignments_checked}/{total_assignments}")
        if solution:
            print(f"  Solution: {solution}")


def main():
    """Main test function"""

    print("MPI BRUTE FORCE SOLVER - BASIC TESTING")
    print("=" * 60)
    print("Note: This test runs without MPI to verify basic functionality")
    print("=" * 60)

    # Test basic solver
    test_brute_force_solver()

    # Test parallel simulation
    test_parallel_simulation()

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nTo run with actual MPI:")
    print("1. Install mpi4py: pip install mpi4py")
    print("2. Run: mpiexec -n 2 python run_mpi_brute_force.py")
    print("3. Or run: python benchmark_mpi_brute_force.py")


if __name__ == "__main__":
    main()
