"""
MPI-Based Parallel Brute Force CNF SAT Solver
Specialized implementation for brute force search with MPI parallelization
"""

import sys
import time
import itertools
import json
from typing import List, Dict, Optional, Any
from mpi4py import MPI

# Import the brute force solver
from Solvers.brute_force_solver import BruteForceSolver


class MPIBruteForceSolver:
    """MPI-based parallel brute force solver with search space partitioning"""

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.local_assignments_checked = 0
        self.total_assignments = 0

    def log(self, message: str) -> None:
        """Rank-aware logging"""
        print(f"[Rank {self.rank}] {message}")
        sys.stdout.flush()

    def solve_parallel(
        self, formula: List[List[str]], variables: List[str]
    ) -> Optional[Dict[str, bool]]:
        """
        Parallel brute force with search space partitioning
        Each process searches a subset of the assignment space
        """
        self.start_time = time.time()

        # Calculate total search space
        self.total_assignments = 2 ** len(variables)
        assignments_per_process = self.total_assignments // self.size
        remainder = self.total_assignments % self.size

        # Calculate assignment range for this process
        start_idx = self.rank * assignments_per_process
        if self.rank < remainder:
            start_idx += self.rank
        else:
            start_idx += remainder

        end_idx = start_idx + assignments_per_process

        if self.rank == 0:
            self.log(f"Starting parallel brute force with {self.size} processes")
            self.log(f"Total assignments: {self.total_assignments}")
            self.log(f"Assignments per process: ~{assignments_per_process}")

        self.log(f"Searching assignments {start_idx} to {end_idx-1}")

        # Create local solver
        solver = BruteForceSolver(formula=formula, variables=variables)

        # Generate all possible assignments
        all_assignments = list(itertools.product([False, True], repeat=len(variables)))

        local_solution = None
        self.local_assignments_checked = 0

        # Search in our assigned range
        for i in range(start_idx, min(end_idx, len(all_assignments))):
            self.local_assignments_checked += 1
            assignment = dict(zip(variables, all_assignments[i]))

            if solver.evaluate_formula(assignment):
                local_solution = assignment
                self.log(f"Solution found: {assignment}")
                break

            # Periodic progress reporting
            if self.local_assignments_checked % 10000 == 0:
                self.log(f"Checked {self.local_assignments_checked} assignments...")

            # Check for early termination signal from other processes
            if self.local_assignments_checked % 1000 == 0:
                if self._check_early_termination():
                    self.log("Early termination signal received")
                    break

        # Gather results from all processes
        all_solutions = self.comm.allgather(local_solution)
        all_assignments_checked = self.comm.allgather(self.local_assignments_checked)

        # Find first valid solution
        global_solution = None
        for sol in all_solutions:
            if sol is not None:
                global_solution = sol
                break

        self.end_time = time.time()

        if self.rank == 0:
            total_checked = sum(all_assignments_checked)
            execution_time = self.end_time - self.start_time
            self.log(f"Parallel brute force complete:")
            self.log(f"  Total assignments checked: {total_checked}")
            self.log(f"  Execution time: {execution_time:.6f}s")
            self.log(f"  Solution found: {global_solution is not None}")
            if global_solution:
                self.log(f"  Solution: {global_solution}")

        return global_solution

    def _check_early_termination(self) -> bool:
        """Check if any other process has found a solution"""
        # Simple implementation: check for incoming messages
        status = MPI.Status()
        if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=999, status=status):
            return True
        return False

    def broadcast_solution(self, solution: Optional[Dict[str, bool]]) -> None:
        """Broadcast solution to all processes to enable early termination"""
        if solution is not None:
            for other_rank in range(self.size):
                if other_rank != self.rank:
                    self.comm.isend(solution, dest=other_rank, tag=999)

    def benchmark_formulas(self, formulas_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark the MPI brute force solver on multiple formulas"""

        if self.rank == 0:
            self.log(
                f"Benchmarking {len(formulas_data)} formulas with {self.size} processes"
            )
            self.log("=" * 60)

        all_results = []

        for i, formula_data in enumerate(formulas_data):
            self.comm.Barrier()  # Synchronize all processes

            if self.rank == 0:
                self.log(f"\nTesting Formula {i+1}/{len(formulas_data)}")
                self.log(f"Formula: {formula_data['formula']}")
                self.log(f"Variables: {formula_data['variables']}")

            # Solve the formula
            solution = self.solve_parallel(
                formula_data["formula"], formula_data["variables"]
            )

            # Collect timing information
            execution_time = (
                self.end_time - self.start_time
                if self.end_time and self.start_time
                else 0
            )
            assignments_checked = sum(
                self.comm.allgather(self.local_assignments_checked)
            )

            if self.rank == 0:
                result = {
                    "formula_id": i,
                    "formula": formula_data["formula"],
                    "variables": formula_data["variables"],
                    "expected_solvable": formula_data.get("expected_solvable", True),
                    "solution_found": solution is not None,
                    "solution": solution,
                    "execution_time": execution_time,
                    "assignments_checked": assignments_checked,
                    "processes": self.size,
                    "total_assignments": self.total_assignments,
                }
                all_results.append(result)

                self.log(f"Result: {'SOLVED' if solution else 'UNSOLVABLE'}")
                self.log(f"Time: {execution_time:.6f}s")
                self.log(
                    f"Assignments checked: {assignments_checked}/{self.total_assignments}"
                )

        return all_results

    def run_performance_analysis(self, formulas_data: List[Dict[str, Any]]) -> None:
        """Run comprehensive performance analysis"""

        if self.rank == 0:
            print("=" * 80)
            print("MPI BRUTE FORCE SOLVER PERFORMANCE ANALYSIS")
            print(f"Running on {self.size} MPI processes")
            print("=" * 80)

        # Run benchmark
        results = self.benchmark_formulas(formulas_data)

        if self.rank == 0:
            # Save results
            filename = f"mpi_brute_force_results_{self.size}proc.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {filename}")

            # Print summary
            print("\n" + "=" * 80)
            print("PERFORMANCE SUMMARY")
            print("=" * 80)

            total_time = sum(r["execution_time"] for r in results)
            total_solved = sum(1 for r in results if r["solution_found"])
            total_assignments = sum(r["assignments_checked"] for r in results)

            print(f"Total formulas tested: {len(results)}")
            print(f"Formulas solved: {total_solved}/{len(results)}")
            print(f"Total execution time: {total_time:.6f}s")
            print(f"Average time per formula: {total_time/len(results):.6f}s")
            print(f"Total assignments checked: {total_assignments:,}")
            print(f"Processes used: {self.size}")

            print("\nDetailed Results:")
            print("-" * 80)
            for result in results:
                status = "✓ SOLVED" if result["solution_found"] else "✗ UNSOLVABLE"
                print(
                    f"Formula {result['formula_id']+1:2d}: {status:12} | "
                    f"Time: {result['execution_time']:8.6f}s | "
                    f"Checked: {result['assignments_checked']:8,}/{result['total_assignments']:8,}"
                )


def create_test_formulas() -> List[Dict[str, Any]]:
    """Create test formulas for benchmarking"""
    return [
        {
            "formula": [["x1", "x2", "x3"], ["-x1", "-x2", "-x3"]],
            "variables": ["x1", "x2", "x3"],
            "expected_solvable": True,
        },
        {
            "formula": [["x1", "x2"], ["-x1", "-x2"], ["x1", "-x2"], ["-x1", "x2"]],
            "variables": ["x1", "x2"],
            "expected_solvable": False,
        },
        {
            "formula": [["x1", "x2", "x3"], ["-x1", "x2", "x3"], ["x1", "-x2", "x3"]],
            "variables": ["x1", "x2", "x3"],
            "expected_solvable": True,
        },
        {
            "formula": [["x1", "x2", "x3", "x4"], ["-x1", "-x2", "-x3", "-x4"]],
            "variables": ["x1", "x2", "x3", "x4"],
            "expected_solvable": True,
        },
        {
            "formula": [
                ["x1", "x2", "x3", "x4", "x5"],
                ["-x1", "-x2", "-x3", "-x4", "-x5"],
                ["x1", "x2", "x3", "x4", "x5"],
            ],
            "variables": ["x1", "x2", "x3", "x4", "x5"],
            "expected_solvable": True,
        },
    ]


def main():
    """Main function to run MPI brute force solver"""

    # Initialize MPI solver
    mpi_solver = MPIBruteForceSolver()

    # Create test formulas
    test_formulas = create_test_formulas()

    # Run performance analysis
    mpi_solver.run_performance_analysis(test_formulas)


if __name__ == "__main__":
    main()
