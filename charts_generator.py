"""
Chart Generation Module for 3-CNF SAT Performance Analysis
Creates professional charts for algorithm comparison and performance analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
import json
import os

# Set style for professional-looking charts
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ChartGenerator:
    """Professional chart generator for CNF SAT performance analysis"""

    def __init__(self, results_file: str = "performance_report.json"):
        """Initialize chart generator with performance data"""
        self.results_file = results_file
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load performance data from JSON file"""
        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Results file {self.results_file} not found. Run performance_analyzer.py first."
            )
            return {}

    def create_comprehensive_report(self, output_dir: str = "charts") -> None:
        """Create comprehensive performance report with multiple charts"""

        if not self.data:
            print("No data available for chart generation.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.data.get("detailed_results", []))

        if df.empty:
            print("No detailed results available for chart generation.")
            return

        # Create various types of charts
        self._create_algorithm_performance_comparison(df, output_dir)
        self._create_scalability_analysis(df, output_dir)
        self._create_problem_size_analysis(df, output_dir)
        self._create_efficiency_heatmap(df, output_dir)
        self._create_success_rate_analysis(df, output_dir)

        print(f"Comprehensive charts generated in {output_dir}/")

    def _create_algorithm_performance_comparison(
        self, df: pd.DataFrame, output_dir: str
    ) -> None:
        """Create detailed algorithm performance comparison"""

        # Filter successful results and exclude parallel for now
        sequential_df = df[
            (df["success"] == True) & (~df["algorithm"].str.contains("parallel"))
        ]

        if sequential_df.empty:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Execution time comparison (box plot)
        sequential_df.boxplot(column="execution_time", by="algorithm", ax=ax1)
        ax1.set_title("Execution Time Distribution by Algorithm")
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_yscale("log")

        # 2. Assignments checked comparison
        sequential_df.boxplot(column="assignments_checked", by="algorithm", ax=ax2)
        ax2.set_title("Assignments Checked Distribution by Algorithm")
        ax2.set_xlabel("Algorithm")
        ax2.set_ylabel("Assignments Checked")
        ax2.set_yscale("log")

        # 3. Memory usage comparison
        sequential_df.boxplot(column="memory_usage", by="algorithm", ax=ax3)
        ax3.set_title("Memory Usage Distribution by Algorithm")
        ax3.set_xlabel("Algorithm")
        ax3.set_ylabel("Memory Usage (MB)")

        # 4. Algorithm efficiency (time vs assignments)
        for algo in sequential_df["algorithm"].unique():
            algo_data = sequential_df[sequential_df["algorithm"] == algo]
            ax4.scatter(
                algo_data["assignments_checked"],
                algo_data["execution_time"],
                label=algo,
                alpha=0.7,
                s=50,
            )

        ax4.set_xlabel("Assignments Checked")
        ax4.set_ylabel("Execution Time (seconds)")
        ax4.set_title("Algorithm Efficiency: Time vs Assignments")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/algorithm_performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_scalability_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create parallel scalability analysis charts"""

        parallel_df = df[df["algorithm"].str.contains("parallel")]

        if parallel_df.empty:
            print("No parallel data available for scalability analysis.")
            return

        # Extract worker count from algorithm name
        parallel_df = parallel_df.copy()
        parallel_df["workers"] = (
            parallel_df["algorithm"].str.extract(r"_(\d+)$").astype(int)
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Execution time vs workers
        workers = parallel_df["workers"].unique()
        workers.sort()
        times = [
            parallel_df[parallel_df["workers"] == w]["execution_time"].mean()
            for w in workers
        ]

        ax1.plot(workers, times, "bo-", linewidth=2, markersize=8)
        ax1.set_title("Parallel Performance Scaling")
        ax1.set_xlabel("Number of Workers")
        ax1.set_ylabel("Average Execution Time (seconds)")
        ax1.grid(True, alpha=0.3)

        # 2. Speedup analysis
        if times and len(times) > 1:
            baseline_time = max(times)  # Use worst performance as baseline
            speedups = [baseline_time / t for t in times]
            ideal_speedups = workers

            ax2.plot(
                workers,
                speedups,
                "ro-",
                linewidth=2,
                markersize=8,
                label="Actual Speedup",
            )
            ax2.plot(workers, ideal_speedups, "g--", linewidth=2, label="Ideal Speedup")
            ax2.set_title("Speedup vs Ideal Speedup")
            ax2.set_xlabel("Number of Workers")
            ax2.set_ylabel("Speedup Factor")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/scalability_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_problem_size_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create analysis of performance vs problem size"""

        successful_df = df[df["success"] == True]

        if successful_df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Execution time vs formula size
        for algo in successful_df["algorithm"].unique():
            if not algo.startswith("parallel"):
                algo_data = successful_df[successful_df["algorithm"] == algo]
                ax1.scatter(
                    algo_data["formula_size"],
                    algo_data["execution_time"],
                    label=algo,
                    alpha=0.7,
                    s=50,
                )

        ax1.set_xlabel("Formula Size (number of clauses)")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_title("Performance vs Problem Size")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Success rate by problem size
        size_success = (
            successful_df.groupby(["formula_size", "algorithm"])
            .size()
            .unstack(fill_value=0)
        )
        total_tests = (
            df.groupby(["formula_size", "algorithm"]).size().unstack(fill_value=0)
        )
        success_rate = (size_success / total_tests * 100).fillna(0)

        success_rate.plot(kind="bar", ax=ax2, width=0.8)
        ax2.set_title("Success Rate by Problem Size")
        ax2.set_xlabel("Formula Size (number of clauses)")
        ax2.set_ylabel("Success Rate (%)")
        ax2.legend(title="Algorithm")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/problem_size_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_efficiency_heatmap(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create efficiency heatmap comparing algorithms"""

        successful_df = df[
            (df["success"] == True) & (~df["algorithm"].str.contains("parallel"))
        ]

        if successful_df.empty:
            return

        # Create efficiency metrics matrix
        metrics = ["execution_time", "assignments_checked", "memory_usage"]
        algorithms = successful_df["algorithm"].unique()

        # Normalize metrics for comparison
        efficiency_data = []
        for algo in algorithms:
            algo_data = successful_df[successful_df["algorithm"] == algo]
            row = []
            for metric in metrics:
                # Use inverse for time and assignments (lower is better)
                if metric in ["execution_time", "assignments_checked"]:
                    value = 1 / (algo_data[metric].mean() + 1e-10)
                else:
                    value = algo_data[metric].mean()
                row.append(value)
            efficiency_data.append(row)

        # Normalize each metric to 0-1 scale
        efficiency_matrix = np.array(efficiency_data)
        for j in range(efficiency_matrix.shape[1]):
            col = efficiency_matrix[:, j]
            if col.max() != col.min():
                efficiency_matrix[:, j] = (col - col.min()) / (col.max() - col.min())

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            efficiency_matrix,
            xticklabels=[
                "Speed\n(1/time)",
                "Efficiency\n(1/assignments)",
                "Memory\nUsage",
            ],
            yticklabels=algorithms,
            annot=True,
            cmap="RdYlGn",
            cbar_kws={"label": "Normalized Performance (higher is better)"},
        )

        plt.title("Algorithm Efficiency Heatmap")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/efficiency_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_success_rate_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create success rate analysis for different problem types"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Success rate by algorithm
        success_by_algo = df.groupby("algorithm")["success"].agg(["count", "sum"])
        success_by_algo["success_rate"] = (
            success_by_algo["sum"] / success_by_algo["count"] * 100
        )

        success_by_algo["success_rate"].plot(kind="bar", ax=ax1, color="skyblue")
        ax1.set_title("Success Rate by Algorithm")
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Success Rate (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # 2. Success rate by satisfiability
        success_by_sat = (
            df.groupby(["algorithm", "is_satisfiable"])["success"].mean() * 100
        )
        success_by_sat.unstack().plot(kind="bar", ax=ax2, width=0.8)
        ax2.set_title("Success Rate: Satisfiable vs Unsatisfiable Problems")
        ax2.set_xlabel("Algorithm")
        ax2.set_ylabel("Success Rate (%)")
        ax2.legend(["Unsatisfiable", "Satisfiable"])
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/success_rate_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main function to generate all charts"""

    print("Generating comprehensive performance charts...")

    chart_generator = ChartGenerator()
    chart_generator.create_comprehensive_report()

    print("Chart generation complete!")


if __name__ == "__main__":
    main()
