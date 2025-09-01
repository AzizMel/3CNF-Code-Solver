"""
Enhanced Main Module for 3-CNF SAT Project
Tests multiple algorithms and demonstrates improvements
"""

import ParallelismBruteForceCNFSolver as CNFSolver
import devTools
import formulas_data


def demo_algorithm_comparison():
    """Demonstrate the performance difference between algorithms"""

    print("=" * 60)
    print("3-CNF SAT ALGORITHM COMPARISON DEMO")
    print("=" * 60)

    # Test on a small sample of formulas
    test_formulas = [
        (0, formulas_data.solvable_formulas[0], True),
        (0, formulas_data.unsolvable_formulas[0], False),
        (1, formulas_data.solvable_formulas[1], True),
    ]

    algorithms = ["brute_force", "dpll", "walksat"]
    variables = ["x1", "x2", "x3"]

    for formula_id, formula, is_solvable in test_formulas:
        print(
            f"\nTesting Formula {formula_id} ({'Solvable' if is_solvable else 'Unsolvable'}):"
        )
        print(f"Formula: {formula}")

        for algorithm in algorithms:
            try:
                solver = CNFSolver.CNFSolver(formula=formula, variables=variables)

                start_time = devTools.run_function_with_timing(solver.solve, algorithm)
                result = start_time[0][0]
                runtime = start_time[0][1]

                print(
                    f"  {algorithm.upper():12} | Time: {runtime:.6f}s | "
                    f"Assignments: {solver.assignments_checked:8d} | "
                    f"Result: {'Found' if result else 'None'}"
                )

            except Exception as e:
                print(f"  {algorithm.upper():12} | ERROR: {str(e)}")

        print("-" * 70)


def run_performance_analysis():
    """Run comprehensive performance analysis"""

    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 60)

    try:
        import performance_analyzer

        analyzer = performance_analyzer.PerformanceAnalyzer()

        print("\nRunning sequential algorithm benchmarks...")
        analyzer.benchmark_sequential_algorithms(max_formula_size=8, runs_per_test=2)

        print("\nRunning parallel algorithm benchmarks...")
        analyzer.benchmark_parallel_approaches(max_workers=4)

        print("\nGenerating performance reports...")
        report = analyzer.generate_performance_report()
        analyzer.export_csv()

        print("\nGenerating performance charts...")
        analyzer.generate_charts()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("Generated files:")
        print("  - performance_report.json")
        print("  - benchmark_results.csv")
        print("  - charts/ directory with visualizations")

    except ImportError as e:
        print(f"Performance analysis requires additional modules: {e}")
        print("Run with virtual environment activated.")


def main():
    """Main function - entry point for the project"""

    print("3-CNF Satisfiability Algorithm Analysis Project")
    print("=" * 60)

    # Demo basic algorithm comparison
    demo_algorithm_comparison()

    # Ask user if they want to run full analysis
    print("\nWould you like to run comprehensive performance analysis? (y/n)")

    # For automated testing, we'll run a limited version
    print("Running limited performance analysis for demonstration...")

    try:
        # Test basic functionality
        formula = formulas_data.solvable_formulas[0]
        variables = ["x1", "x2", "x3"]

        solver = CNFSolver.CNFSolver(formula=formula, variables=variables)

        print(f"\nTesting enhanced solver on formula: {formula}")

        for algo in ["brute_force", "dpll"]:
            result = solver.solve(algo)
            print(
                f"{algo.upper()}: Found solution = {result is not None}, "
                f"Assignments checked = {solver.assignments_checked}"
            )

    except Exception as e:
        print(f"Error in basic testing: {e}")


if __name__ == "__main__":
    main()
