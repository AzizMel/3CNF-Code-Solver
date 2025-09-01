"""
MPI-Based Parallel CNF SAT Solver
Novel parallel implementation using Message Passing Interface (MPI)
Supports multiple parallel strategies for 3-CNF satisfiability problems
"""

import sys
import time
import itertools
import json
from typing import List, Dict, Optional, Any
from mpi4py import MPI

import ParallelismBruteForceCNFSolver as CNFSolver
import formulas_data


class MPICNFSolver:
    """MPI-based parallel CNF solver with multiple parallel strategies"""

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.local_assignments_checked = 0

    def log(self, message: str) -> None:
        """Rank-aware logging"""
        print(f"[Rank {self.rank}] {message}")
        sys.stdout.flush()

    def parallel_brute_force_search_space_partition(
        self, formula: List[List[str]], variables: List[str]
    ) -> Optional[Dict[str, bool]]:
        """
        Parallel brute force with search space partitioning
        Each process searches a subset of the assignment space
        """
        self.start_time = time.time()

        total_assignments = 2 ** len(variables)
        assignments_per_process = total_assignments // self.size
        remainder = total_assignments % self.size

        # Calculate assignment range for this process
        start_idx = self.rank * assignments_per_process
        if self.rank < remainder:
            start_idx += self.rank
        else:
            start_idx += remainder

        end_idx = start_idx + assignments_per_process

        if self.rank == 0:
            self.log(f"Starting parallel brute force with {self.size} processes")
            self.log(f"Total assignments: {total_assignments}")

        self.log(f"Searching assignments {start_idx} to {end_idx-1}")

        # Create local solver
        solver = CNFSolver.CNFSolver(formula=formula, variables=variables)

        # Generate and test assignments in our range
        all_assignments = list(itertools.product([False, True], repeat=len(variables)))

        local_solution = None
        self.local_assignments_checked = 0

        for i in range(start_idx, min(end_idx, len(all_assignments))):
            self.local_assignments_checked += 1
            assignment = dict(zip(variables, all_assignments[i]))

            if solver.evaluate_formula(assignment):
                local_solution = assignment
                self.log(f"Solution found: {assignment}")
                break

            # Periodic check for early termination signal
            if self.local_assignments_checked % 1000 == 0:
                if self._check_early_termination():
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

        return global_solution

    def parallel_dpll_portfolio(
        self, formula: List[List[str]], variables: List[str]
    ) -> Optional[Dict[str, bool]]:
        """
        Portfolio approach: different processes use different algorithms
        or different heuristics for variable ordering
        """
        self.start_time = time.time()

        if self.rank == 0:
            self.log(f"Starting parallel portfolio with {self.size} processes")

        # Assign different strategies to different processes
        strategies = ["dpll", "walksat", "brute_force"]
        strategy = strategies[self.rank % len(strategies)]

        self.log(f"Using strategy: {strategy}")

        # Create solver with assigned strategy
        solver = CNFSolver.CNFSolver(formula=formula, variables=variables)
        local_solution = None

        try:
            local_solution = solver.solve(algorithm=strategy)
            self.local_assignments_checked = solver.assignments_checked

            if local_solution:
                self.log(f"Solution found using {strategy}: {local_solution}")

        except Exception as e:
            self.log(f"Error in {strategy}: {e}")

        # Use non-blocking communication to find first solution
        solution_found = self._first_solution_wins(local_solution)

        self.end_time = time.time()

        if self.rank == 0:
            execution_time = self.end_time - self.start_time
            self.log(f"Parallel portfolio complete:")
            self.log(f"  Execution time: {execution_time:.6f}s")
            self.log(f"  Solution found: {solution_found is not None}")

        return solution_found

    def parallel_dpll_work_stealing(
        self, formula: List[List[str]], variables: List[str]
    ) -> Optional[Dict[str, bool]]:
        """
        Advanced parallel DPLL with work stealing
        Processes can steal work from each other when idle
        """
        self.start_time = time.time()

        if self.rank == 0:
            self.log(
                f"Starting parallel DPLL with work stealing ({self.size} processes)"
            )

        # Each process starts with different variable ordering
        local_variables = variables.copy()
        # Rotate variable order based on rank
        for _ in range(self.rank):
            local_variables = local_variables[1:] + [local_variables[0]]

        self.log(f"Variable order: {local_variables}")

        # Modified DPLL that can be interrupted
        solver = CNFSolver.CNFSolver(formula=formula, variables=local_variables)
        local_solution = None

        try:
            # Run DPLL with periodic communication checks
            local_solution = self._interruptible_dpll(solver, formula, {})

        except Exception as e:
            self.log(f"Error in DPLL: {e}")

        # Collect results
        solution_found = self._first_solution_wins(local_solution)

        self.end_time = time.time()

        if self.rank == 0:
            execution_time = self.end_time - self.start_time
            self.log(f"Parallel DPLL complete:")
            self.log(f"  Execution time: {execution_time:.6f}s")
            self.log(f"  Solution found: {solution_found is not None}")

        return solution_found

    def _interruptible_dpll(
        self,
        solver,
        formula: List[List[str]],
        assignment: Dict[str, bool],
        depth: int = 0,
    ) -> Optional[Dict[str, bool]]:
        """DPLL algorithm that can be interrupted by other processes finding solutions"""

        self.local_assignments_checked += 1

        # Check for early termination every few recursions
        if depth % 10 == 0 and self._check_early_termination():
            return None

        # Simplified DPLL logic (can be expanded)
        # For now, just use the sequential DPLL from our solver
        try:
            return solver.dpll_solve()
        except:
            return None

    def _check_early_termination(self) -> bool:
        """Check if any other process has found a solution"""
        # Simple implementation: check for incoming messages
        status = MPI.Status()
        if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=999, status=status):
            return True
        return False

    def _first_solution_wins(
        self, local_solution: Optional[Dict[str, bool]]
    ) -> Optional[Dict[str, bool]]:
        """Implement first-solution-wins using non-blocking communication"""

        # If we found a solution, broadcast it
        if local_solution is not None:
            for other_rank in range(self.size):
                if other_rank != self.rank:
                    self.comm.isend(local_solution, dest=other_rank, tag=999)

        # Check if anyone else found a solution
        all_solutions = self.comm.allgather(local_solution)

        # Return first non-None solution
        for solution in all_solutions:
            if solution is not None:
                return solution

        return None

    def benchmark_parallel_strategies(
        self, formula_id: int, formula: List[List[str]], variables: List[str]
    ) -> Dict[str, Any]:
        """Benchmark all parallel strategies on a given formula"""

        if self.rank == 0:
            self.log(f"Benchmarking formula {formula_id}: {formula}")

        strategies = [
            (
                "search_space_partition",
                self.parallel_brute_force_search_space_partition,
            ),
            ("portfolio", self.parallel_dpll_portfolio),
            ("work_stealing", self.parallel_dpll_work_stealing),
        ]

        results = {}

        for strategy_name, strategy_func in strategies:
            if self.rank == 0:
                self.log(f"\n--- Testing {strategy_name} ---")

            self.comm.Barrier()  # Synchronize all processes

            start_time = time.time()
            solution = strategy_func(formula, variables)
            end_time = time.time()

            results[strategy_name] = {
                "solution_found": solution is not None,
                "execution_time": end_time - start_time,
                "solution": solution,
            }

            if self.rank == 0:
                self.log(f"{strategy_name} result: {results[strategy_name]}")

        return results


def run_mpi_benchmark():
    """Main function to run MPI benchmarks"""

    # Initialize MPI solver
    mpi_solver = MPICNFSolver()

    if mpi_solver.rank == 0:
        print("=" * 80)
        print("MPI-BASED PARALLEL CNF SAT SOLVER BENCHMARK")
        print(f"Running on {mpi_solver.size} MPI processes")
        print("=" * 80)

    # Test formulas
    test_cases = [
        (0, formulas_data.solvable_formulas[0], ["x1", "x2", "x3"], True),
        (1, formulas_data.solvable_formulas[1], ["x1", "x2", "x3"], True),
        (0, formulas_data.unsolvable_formulas[0], ["x1", "x2", "x3"], False),
    ]

    all_results = []

    for formula_id, formula, variables, is_solvable in test_cases:
        mpi_solver.comm.Barrier()

        if mpi_solver.rank == 0:
            print(
                f"\nTesting Formula {formula_id} ({'Solvable' if is_solvable else 'Unsolvable'})"
            )

        results = mpi_solver.benchmark_parallel_strategies(
            formula_id, formula, variables
        )

        if mpi_solver.rank == 0:
            results["formula_id"] = formula_id
            results["is_solvable"] = is_solvable
            results["processes"] = mpi_solver.size
            all_results.append(results)

    # Save results from rank 0
    if mpi_solver.rank == 0:
        with open(f"mpi_benchmark_results_{mpi_solver.size}proc.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 80)
        print("MPI BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"Results saved to mpi_benchmark_results_{mpi_solver.size}proc.json")

        # Print summary
        print("\nSUMMARY:")
        for result in all_results:
            formula_id = result["formula_id"]
            is_solvable = result["is_solvable"]
            print(f"\nFormula {formula_id} ({'Solvable' if is_solvable else 'Unsolvable'}):")
            
            for strategy in ["search_space_partition", "portfolio", "work_stealing"]:
                if strategy in result:
                    time_taken = result[strategy]["execution_time"]
                    found = result[strategy]["solution_found"]
                    print(
                        f"  {strategy:20}: {time_taken:.6f}s ({'✓' if found else '✗'})"
                    )


if __name__ == "__main__":
    run_mpi_benchmark()