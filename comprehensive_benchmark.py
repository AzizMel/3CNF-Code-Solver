"""
Comprehensive Benchmarking Framework for 3-CNF SAT Project
Runs all phases of performance analysis including MPI scaling tests
"""

import os
import subprocess
import json
import time
import multiprocessing as mp
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

import performance_analyzer
import charts_generator


class ComprehensiveBenchmark:
    """Master benchmark runner for all project phases"""
    
    def __init__(self, max_mpi_processes: int = None):
        self.max_mpi_processes = max_mpi_processes or min(mp.cpu_count(), 8)
        self.results_dir = "benchmark_results"
        self.charts_dir = "comprehensive_charts"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        self.all_results = {
            "sequential": {},
            "multiprocessing": {},
            "mpi": {},
            "scaling_analysis": {},
            "timestamp": time.time()
        }
    
    def run_phase1_sequential_analysis(self) -> None:
        """Phase 1: Comprehensive sequential algorithm analysis"""
        print("\n" + "="*60)
        print("PHASE 1: SEQUENTIAL ALGORITHM ANALYSIS")
        print("="*60)
        
        analyzer = performance_analyzer.PerformanceAnalyzer()
        
        # Test with various problem sizes
        print("Running sequential benchmarks...")
        analyzer.benchmark_sequential_algorithms(max_formula_size=12, runs_per_test=5)
        
        # Generate detailed reports
        report = analyzer.generate_performance_report(
            f"{self.results_dir}/sequential_performance_report.json"
        )
        
        analyzer.export_csv(f"{self.results_dir}/sequential_benchmark_results.csv")
        analyzer.generate_charts(f"{self.charts_dir}/sequential")
        
        self.all_results["sequential"] = report
        print("Phase 1 complete: Sequential analysis saved")
    
    def run_phase2_multiprocessing_analysis(self) -> None:
        """Phase 2: Multiprocessing parallel analysis"""
        print("\n" + "="*60)
        print("PHASE 2: MULTIPROCESSING PARALLEL ANALYSIS")
        print("="*60)
        
        analyzer = performance_analyzer.PerformanceAnalyzer()
        
        # Test different worker counts
        worker_counts = [1, 2, 4, min(8, mp.cpu_count()), mp.cpu_count()]
        worker_counts = sorted(list(set(worker_counts)))  # Remove duplicates
        
        print(f"Testing with worker counts: {worker_counts}")
        
        for workers in worker_counts:
            print(f"\nTesting with {workers} workers...")
            analyzer.benchmark_parallel_approaches(max_workers=workers)
        
        # Generate reports
        report = analyzer.generate_performance_report(
            f"{self.results_dir}/multiprocessing_performance_report.json"
        )
        
        analyzer.export_csv(f"{self.results_dir}/multiprocessing_benchmark_results.csv")
        analyzer.generate_charts(f"{self.charts_dir}/multiprocessing")
        
        self.all_results["multiprocessing"] = report
        print("Phase 2 complete: Multiprocessing analysis saved")
    
    def run_phase3_mpi_analysis(self) -> None:
        """Phase 3: MPI parallel analysis"""
        print("\n" + "="*60)
        print("PHASE 3: MPI PARALLEL ANALYSIS")
        print("="*60)
        
        # Test different MPI process counts
        process_counts = [1, 2, 4, min(8, self.max_mpi_processes), self.max_mpi_processes]
        process_counts = sorted(list(set([p for p in process_counts if p > 0])))
        
        print(f"Testing MPI with process counts: {process_counts}")
        
        mpi_results = {}
        
        for proc_count in process_counts:
            print(f"\nRunning MPI with {proc_count} processes...")
            
            try:
                # Run MPI benchmark
                cmd = [
                    "mpirun", "-n", str(proc_count), 
                    "python", "mpi_cnf_solver.py"
                ]
                
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    print(f"  MPI {proc_count} processes: Success ({end_time - start_time:.2f}s)")
                    
                    # Load results if available
                    result_file = f"mpi_benchmark_results_{proc_count}proc.json"
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            mpi_results[proc_count] = json.load(f)
                        
                        # Move to results directory
                        os.rename(result_file, f"{self.results_dir}/{result_file}")
                else:
                    print(f"  MPI {proc_count} processes: Failed")
                    print(f"  Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"  MPI {proc_count} processes: Timeout")
            except Exception as e:
                print(f"  MPI {proc_count} processes: Error - {e}")
        
        self.all_results["mpi"] = mpi_results
        print("Phase 3 complete: MPI analysis saved")
    
    def run_phase4_scaling_analysis(self) -> None:
        """Phase 4: Comprehensive scaling analysis"""
        print("\n" + "="*60)
        print("PHASE 4: SCALING ANALYSIS")
        print("="*60)
        
        # Analyze scaling from all collected data
        scaling_data = self._extract_scaling_data()
        
        # Generate scaling charts
        self._create_scaling_charts(scaling_data)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(scaling_data)
        
        self.all_results["scaling_analysis"] = {
            "scaling_data": scaling_data,
            "efficiency_metrics": efficiency_metrics
        }
        
        print("Phase 4 complete: Scaling analysis generated")
    
    def _extract_scaling_data(self) -> Dict[str, List]:
        """Extract scaling data from all benchmark results"""
        scaling_data = {
            "processors": [],
            "multiprocessing_times": [],
            "mpi_times": [],
            "sequential_baseline": None
        }
        
        # Get sequential baseline
        seq_results = self.all_results.get("sequential", {})
        if "algorithm_performance" in seq_results:
            # Use DPLL as baseline (most practical algorithm)
            if "dpll" in seq_results["algorithm_performance"]:
                scaling_data["sequential_baseline"] = seq_results["algorithm_performance"]["dpll"]["avg_execution_time"]
        
        # Extract multiprocessing data
        mp_results = self.all_results.get("multiprocessing", {})
        if "detailed_results" in mp_results:
            mp_by_workers = {}
            for result in mp_results["detailed_results"]:
                if result["algorithm"].startswith("parallel_brute_force_"):
                    workers = int(result["algorithm"].split("_")[-1])
                    if workers not in mp_by_workers:
                        mp_by_workers[workers] = []
                    mp_by_workers[workers].append(result["execution_time"])
            
            for workers in sorted(mp_by_workers.keys()):
                avg_time = sum(mp_by_workers[workers]) / len(mp_by_workers[workers])
                scaling_data["processors"].append(workers)
                scaling_data["multiprocessing_times"].append(avg_time)
        
        # Extract MPI data
        mpi_results = self.all_results.get("mpi", {})
        mpi_times = []
        for proc_count in sorted(mpi_results.keys()):
            if isinstance(mpi_results[proc_count], list) and mpi_results[proc_count]:
                # Average execution time across strategies
                avg_times = []
                for test_result in mpi_results[proc_count]:
                    for strategy in ["search_space_partition", "portfolio", "work_stealing"]:
                        if strategy in test_result:
                            avg_times.append(test_result[strategy]["execution_time"])
                
                if avg_times:
                    avg_time = sum(avg_times) / len(avg_times)
                    if len(mpi_times) < len(scaling_data["processors"]):
                        mpi_times.append(avg_time)
        
        # Pad mpi_times to match processor counts
        while len(mpi_times) < len(scaling_data["processors"]):
            mpi_times.append(None)
        
        scaling_data["mpi_times"] = mpi_times
        
        return scaling_data
    
    def _create_scaling_charts(self, scaling_data: Dict[str, List]) -> None:
        """Create comprehensive scaling analysis charts"""
        
        if not scaling_data["processors"]:
            print("No scaling data available for chart generation")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        processors = scaling_data["processors"]
        mp_times = scaling_data["multiprocessing_times"]
        mpi_times = scaling_data["mpi_times"]
        
        # 1. Execution time vs processors
        ax1.plot(processors, mp_times, 'bo-', label='Multiprocessing', linewidth=2, markersize=8)
        valid_mpi = [(p, t) for p, t in zip(processors, mpi_times) if t is not None]
        if valid_mpi:
            mpi_proc, mpi_t = zip(*valid_mpi)
            ax1.plot(mpi_proc, mpi_t, 'ro-', label='MPI', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Number of Processors')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Parallel Performance Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup analysis
        baseline = scaling_data["sequential_baseline"]
        if baseline and baseline > 0:
            mp_speedups = [baseline / t for t in mp_times]
            ax2.plot(processors, mp_speedups, 'bo-', label='Multiprocessing Speedup', linewidth=2)
            
            if valid_mpi:
                mpi_speedups = [baseline / t for _, t in valid_mpi]
                ax2.plot(mpi_proc, mpi_speedups, 'ro-', label='MPI Speedup', linewidth=2)
            
            # Ideal speedup line
            ax2.plot(processors, processors, 'g--', label='Ideal Speedup', linewidth=2)
            
            ax2.set_xlabel('Number of Processors')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Speedup vs Ideal Speedup')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency analysis
        if baseline and baseline > 0:
            mp_efficiency = [speedup / proc for speedup, proc in zip(mp_speedups, processors)]
            ax3.plot(processors, mp_efficiency, 'bo-', label='Multiprocessing Efficiency', linewidth=2)
            
            if valid_mpi:
                mpi_efficiency = [speedup / proc for speedup, proc in zip(mpi_speedups, mpi_proc)]
                ax3.plot(mpi_proc, mpi_efficiency, 'ro-', label='MPI Efficiency', linewidth=2)
            
            ax3.axhline(y=1.0, color='g', linestyle='--', label='Perfect Efficiency')
            ax3.set_xlabel('Number of Processors')
            ax3.set_ylabel('Parallel Efficiency')
            ax3.set_title('Parallel Efficiency Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.2)
        
        # 4. Algorithm comparison (if we have multiple strategies)
        if "mpi" in self.all_results and self.all_results["mpi"]:
            strategies = ["search_space_partition", "portfolio", "work_stealing"]
            strategy_times = {strategy: [] for strategy in strategies}
            proc_counts = []
            
            for proc_count, results in self.all_results["mpi"].items():
                if isinstance(results, list) and results:
                    proc_counts.append(proc_count)
                    for strategy in strategies:
                        times = []
                        for test_result in results:
                            if strategy in test_result:
                                times.append(test_result[strategy]["execution_time"])
                        avg_time = sum(times) / len(times) if times else 0
                        strategy_times[strategy].append(avg_time)
            
            for strategy in strategies:
                if strategy_times[strategy]:
                    ax4.plot(proc_counts, strategy_times[strategy], 
                            'o-', label=strategy.replace('_', ' ').title(), linewidth=2)
            
            ax4.set_xlabel('Number of Processors')
            ax4.set_ylabel('Execution Time (seconds)')
            ax4.set_title('MPI Strategy Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.charts_dir}/comprehensive_scaling_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scaling charts saved to {self.charts_dir}/comprehensive_scaling_analysis.png")
    
    def _calculate_efficiency_metrics(self, scaling_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate comprehensive efficiency metrics"""
        
        metrics = {
            "multiprocessing_metrics": {},
            "mpi_metrics": {},
            "comparison": {}
        }
        
        baseline = scaling_data["sequential_baseline"]
        if not baseline or baseline <= 0:
            return metrics
        
        processors = scaling_data["processors"]
        
        # Multiprocessing metrics
        mp_times = scaling_data["multiprocessing_times"]
        if mp_times:
            mp_speedups = [baseline / t for t in mp_times]
            mp_efficiencies = [s / p for s, p in zip(mp_speedups, processors)]
            
            metrics["multiprocessing_metrics"] = {
                "max_speedup": max(mp_speedups),
                "max_efficiency": max(mp_efficiencies),
                "avg_efficiency": sum(mp_efficiencies) / len(mp_efficiencies),
                "scalability_factor": mp_speedups[-1] / mp_speedups[0] if len(mp_speedups) > 1 else 1.0
            }
        
        # MPI metrics
        mpi_times = [t for t in scaling_data["mpi_times"] if t is not None]
        if mpi_times:
            valid_procs = processors[:len(mpi_times)]
            mpi_speedups = [baseline / t for t in mpi_times]
            mpi_efficiencies = [s / p for s, p in zip(mpi_speedups, valid_procs)]
            
            metrics["mpi_metrics"] = {
                "max_speedup": max(mpi_speedups),
                "max_efficiency": max(mpi_efficiencies),
                "avg_efficiency": sum(mpi_efficiencies) / len(mpi_efficiencies),
                "scalability_factor": mpi_speedups[-1] / mpi_speedups[0] if len(mpi_speedups) > 1 else 1.0
            }
        
        return metrics
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE FINAL REPORT")
        print("="*60)
        
        # Save all results
        with open(f"{self.results_dir}/comprehensive_benchmark_results.json", 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Generate all charts
        chart_gen = charts_generator.ChartGenerator(f"{self.results_dir}/sequential_performance_report.json")
        chart_gen.create_comprehensive_report(f"{self.charts_dir}/final_charts")
        
        print(f"Comprehensive report saved to {self.results_dir}/")
        print(f"All charts saved to {self.charts_dir}/")
    
    def _generate_summary_report(self) -> None:
        """Generate human-readable summary report"""
        
        summary_lines = [
            "3-CNF SATISFIABILITY ALGORITHM ANALYSIS",
            "COMPREHENSIVE BENCHMARK RESULTS",
            "=" * 60,
            "",
            f"Benchmark completed at: {time.ctime(self.all_results['timestamp'])}",
            f"System info: {mp.cpu_count()} CPU cores available",
            ""
        ]
        
        # Sequential results summary
        if "sequential" in self.all_results and "algorithm_performance" in self.all_results["sequential"]:
            summary_lines.extend([
                "SEQUENTIAL ALGORITHM PERFORMANCE:",
                "-" * 40
            ])
            
            for algo, stats in self.all_results["sequential"]["algorithm_performance"].items():
                summary_lines.extend([
                    f"{algo.upper()}:",
                    f"  Average execution time: {stats['avg_execution_time']:.6f}s",
                    f"  Average assignments checked: {stats['avg_assignments_checked']:.0f}",
                    f"  Test count: {stats['test_count']}",
                    ""
                ])
        
        # Scaling analysis summary
        if "scaling_analysis" in self.all_results and "efficiency_metrics" in self.all_results["scaling_analysis"]:
            metrics = self.all_results["scaling_analysis"]["efficiency_metrics"]
            
            summary_lines.extend([
                "PARALLEL PERFORMANCE SUMMARY:",
                "-" * 40
            ])
            
            if "multiprocessing_metrics" in metrics and metrics["multiprocessing_metrics"]:
                mp_metrics = metrics["multiprocessing_metrics"]
                summary_lines.extend([
                    "Multiprocessing:",
                    f"  Maximum speedup: {mp_metrics.get('max_speedup', 0):.2f}x",
                    f"  Maximum efficiency: {mp_metrics.get('max_efficiency', 0):.2f}",
                    f"  Average efficiency: {mp_metrics.get('avg_efficiency', 0):.2f}",
                    ""
                ])
            
            if "mpi_metrics" in metrics and metrics["mpi_metrics"]:
                mpi_metrics = metrics["mpi_metrics"]
                summary_lines.extend([
                    "MPI:",
                    f"  Maximum speedup: {mpi_metrics.get('max_speedup', 0):.2f}x",
                    f"  Maximum efficiency: {mpi_metrics.get('max_efficiency', 0):.2f}",
                    f"  Average efficiency: {mpi_metrics.get('avg_efficiency', 0):.2f}",
                    ""
                ])
        
        # Save summary
        with open(f"{self.results_dir}/benchmark_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Print summary
        print('\n'.join(summary_lines))


def main():
    """Main function to run comprehensive benchmark"""
    
    print("3-CNF SAT COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)
    print("This will run all phases of the project analysis:")
    print("1. Sequential algorithm analysis")
    print("2. Multiprocessing parallel analysis") 
    print("3. MPI parallel analysis")
    print("4. Comprehensive scaling analysis")
    print("5. Final report generation")
    print("=" * 60)
    
    # Initialize benchmark suite
    benchmark = ComprehensiveBenchmark(max_mpi_processes=8)
    
    try:
        # Run all phases
        benchmark.run_phase1_sequential_analysis()
        benchmark.run_phase2_multiprocessing_analysis()
        benchmark.run_phase3_mpi_analysis()
        benchmark.run_phase4_scaling_analysis()
        benchmark.generate_final_report()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE BENCHMARK COMPLETE!")
        print("="*60)
        print("Check the following directories for results:")
        print(f"  - {benchmark.results_dir}/ for data files")
        print(f"  - {benchmark.charts_dir}/ for visualizations")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()