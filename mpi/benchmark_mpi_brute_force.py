#!/usr/bin/env python3
"""
Comprehensive benchmark script for MPI brute force solver
Tests different numbers of processes and formula sizes
"""

import subprocess
import json
import time
import os
from typing import List, Dict, Any


def run_mpi_benchmark(num_processes: int, formula_size: int) -> Dict[str, Any]:
    """Run MPI benchmark with specified number of processes and formula size"""

    print(
        f"Running benchmark with {num_processes} processes, formula size {formula_size}"
    )

    # Create a temporary script for this specific test
    script_content = f'''
import sys
sys.path.append(".")
from mpi_brute_force_solver import MPIBruteForceSolver
import itertools

def create_large_formula(size):
    """Create a formula with specified number of variables"""
    variables = [f"x{{i+1}}" for i in range(size)]
    
    # Create a solvable formula
    formula = []
    for i in range(size):
        clause = []
        for j in range(size):
            if i == j:
                clause.append(f"x{{j+1}}")
            else:
                clause.append(f"-x{{j+1}}")
        formula.append(clause)
    
    return {{
        "formula": formula,
        "variables": variables,
        "expected_solvable": True
    }}

# Initialize solver
mpi_solver = MPIBruteForceSolver()

# Create test formula
test_formula = create_large_formula({formula_size})
formulas = [test_formula]

# Run benchmark
results = mpi_solver.benchmark_formulas(formulas)

if mpi_solver.rank == 0:
    # Save results
    filename = f"mpi_brute_force_benchmark_{{mpi_solver.size}}proc_size{formula_size}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {{filename}}")
    
    # Print summary
    for result in results:
        print(f"Formula size {{formula_size}}: {{'SOLVED' if result['solution_found'] else 'UNSOLVABLE'}}")
        print(f"Time: {{result['execution_time']:.6f}}s")
        print(f"Assignments checked: {{result['assignments_checked']:,}}/{{result['total_assignments']:,}}")
        print(f"Speedup: {{result['total_assignments'] / result['execution_time']:.0f}} assignments/sec")
'''

    # Write temporary script
    temp_script = f"temp_benchmark_{num_processes}_{formula_size}.py"
    with open(temp_script, "w") as f:
        f.write(script_content)

    try:
        # Run MPI command
        cmd = ["mpiexec", "-n", str(num_processes), "python", temp_script]
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 minute timeout
        end_time = time.time()

        if result.returncode == 0:
            print(f"✓ Benchmark completed successfully")
            print(f"  Total time: {end_time - start_time:.2f}s")
            print(f"  Output: {result.stdout}")
        else:
            print(f"✗ Benchmark failed")
            print(f"  Error: {result.stderr}")

        return {
            "num_processes": num_processes,
            "formula_size": formula_size,
            "success": result.returncode == 0,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        print(f"✗ Benchmark timed out after 5 minutes")
        return {
            "num_processes": num_processes,
            "formula_size": formula_size,
            "success": False,
            "execution_time": 300,
            "timeout": True,
        }
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    """Run comprehensive MPI brute force benchmarks"""

    print("=" * 80)
    print("MPI BRUTE FORCE SOLVER COMPREHENSIVE BENCHMARK")
    print("=" * 80)

    # Test configurations
    process_counts = [1, 2, 4, 8]  # Number of MPI processes
    formula_sizes = [3, 4, 5]  # Number of variables in formulas

    all_results = []

    for formula_size in formula_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING FORMULA SIZE: {formula_size} VARIABLES")
        print(f"{'='*60}")

        for num_processes in process_counts:
            print(f"\n--- Testing {num_processes} processes ---")
            result = run_mpi_benchmark(num_processes, formula_size)
            all_results.append(result)

            # Small delay between tests
            time.sleep(1)

    # Save all results
    with open("comprehensive_mpi_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to comprehensive_mpi_benchmark_results.json")

    # Print summary
    print("\nSUMMARY:")
    print("-" * 80)
    for result in all_results:
        status = "✓" if result["success"] else "✗"
        timeout = " (TIMEOUT)" if result.get("timeout") else ""
        print(
            f"{status} {result['num_processes']:2d} processes, "
            f"size {result['formula_size']}: "
            f"{result['execution_time']:6.2f}s{timeout}"
        )


if __name__ == "__main__":
    main()
