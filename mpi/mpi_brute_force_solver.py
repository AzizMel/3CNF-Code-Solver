import sys
import time
import itertools
import json
import os
from typing import List, Dict, Optional, Any
from mpi4py import MPI
import formulas_data
# to run: mpiexec -n 2 ./venv/bin/python mpi/run_mpi_brute_force.py 

# Add the parent directory to the Python path to find the Solvers module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

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
        solver = BruteForceSolver(formula=formula)
        solver.variables = variables

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

    def solve_single_formula(self, formula_id: int, formula: List[List[str]], variables: List[str]) -> Dict[str, Any]:
        """Solve a single formula and return detailed results"""
        self.comm.Barrier()  # Synchronize all processes
        
        if self.rank == 0:
            self.log(f"Solving Formula {formula_id}")
            self.log(f"Formula: {formula}")
            self.log(f"Variables: {variables}")

        # Solve the formula
        solution = self.solve_parallel(formula, variables)

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
                "formula_id": formula_id,
                "formula_to_solve": formula,
                "len_formula": len(formula),
                "mpi_brute_force_solver": {
                    "time": execution_time,  # Python will automatically use scientific notation for small numbers
                    "solved": solution is not None,
                    "solution": solution,
                    "assignments_checked": assignments_checked,
                    "processes": self.size,
                    "total_assignments": self.total_assignments,
                }
            }
            
            self.log(f"Result: {'SOLVED' if solution else 'UNSOLVABLE'}")
            self.log(f"Time: {execution_time:.6f}s")
            self.log(f"Assignments checked: {assignments_checked}/{self.total_assignments}")
            
            return result
        
        return None

    def benchmark_formulas(self, formulas_data: Dict[int, List[List[str]]]) -> Dict[str, Any]:
        """Benchmark the MPI brute force solver on multiple formulas"""

        if self.rank == 0:
            self.log(
                f"Benchmarking {len(formulas_data)} formulas with {self.size} processes"
            )
            self.log("=" * 60)

        all_results = {}

        for formula_id, formula in formulas_data.items():
            # Extract variables from the formula
            variables = set()
            for clause in formula:
                for literal in clause:
                    var = literal.lstrip('-')
                    variables.add(var)
            variables = sorted(list(variables))

            result = self.solve_single_formula(formula_id, formula, variables)
            if result is not None:
                all_results[str(formula_id)] = result

        return all_results

    def run_performance_analysis(self) -> None:
        """Run comprehensive performance analysis"""

        if self.rank == 0:
            print("=" * 80)
            print("MPI BRUTE FORCE SOLVER PERFORMANCE ANALYSIS")
            print(f"Running on {self.size} MPI processes")
            print("=" * 80)

        # Run benchmark
        results = self.benchmark_formulas(formulas_data.formulas)

        if self.rank == 0:
            # Save results to Results/MPIAnalysis folder
            os.makedirs("Results/MPIAnalysis", exist_ok=True)
            filename = f"Results/MPIAnalysis/mpi_brute_force_results_{self.size}proc.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {filename}")

            # Print summary
            print("\n" + "=" * 80)
            print("PERFORMANCE SUMMARY")
            print("=" * 80)

            total_time = sum(r["mpi_brute_force_solver"]["time"] for r in results.values())
            total_solved = sum(1 for r in results.values() if r["mpi_brute_force_solver"]["solved"])
            total_assignments = sum(r["mpi_brute_force_solver"]["assignments_checked"] for r in results.values())

            print(f"Total formulas tested: {len(results)}")
            print(f"Formulas solved: {total_solved}/{len(results)}")
            print(f"Total execution time: {total_time:.6f}s")
            print(f"Average time per formula: {total_time/len(results):.6f}s")
            print(f"Total assignments checked: {total_assignments:,}")
            print(f"Processes used: {self.size}")

            print("\nDetailed Results:")
            print("-" * 80)
            for formula_id, result in results.items():
                status = "✓ SOLVED" if result["mpi_brute_force_solver"]["solved"] else "✗ UNSOLVABLE"
                print(
                    f"Formula {formula_id:2s}: {status:12} | "
                    f"Time: {result['mpi_brute_force_solver']['time']:8.6f}s | "
                    f"Checked: {result['mpi_brute_force_solver']['assignments_checked']:8,}/{result['mpi_brute_force_solver']['total_assignments']:8,}"
                )


def main():
    """Main function to run MPI brute force solver"""

    # Initialize MPI solver
    mpi_solver = MPIBruteForceSolver()
    mpi_solver.run_performance_analysis()


if __name__ == "__main__":
    main()
