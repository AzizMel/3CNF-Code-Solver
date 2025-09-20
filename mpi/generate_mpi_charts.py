#!/usr/bin/env python3
"""
MPI Results Chart Generator
Creates comprehensive charts and visualizations from MPI brute force solver results
Usage: python generate_mpi_charts.py
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import formulas_data


def load_mpi_results() -> Dict[str, Any]:
    """Load MPI comprehensive results"""
    results_file = "Results/MPIAnalysis/mpi_comprehensive_scaling_report.json"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        print("Please run run_mpi_scaling_analysis.py first to generate results.")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_performance_scaling_chart(results: Dict[str, Any]) -> None:
    """Create performance scaling chart"""
    
    print("Creating performance scaling chart...")
    
    performance = results["performance_analysis"]
    process_counts = [int(p) for p in performance["average_times_per_process"].keys()]
    process_counts.sort()
    
    times = [performance["average_times_per_process"][str(p)] for p in process_counts]
    
    plt.figure(figsize=(12, 8))
    
    # Plot execution time vs processes
    plt.subplot(2, 2, 1)
    plt.plot(process_counts, times, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Average Time per Formula (seconds)')
    plt.title('MPI Execution Time vs Number of Processes')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot speedup
    if "speedup_analysis" in performance:
        speedup_processes = [int(p) for p in performance["speedup_analysis"].keys()]
        speedup_processes.sort()
        speedups = [performance["speedup_analysis"][str(p)] for p in speedup_processes]
        
        plt.subplot(2, 2, 2)
        plt.plot(speedup_processes, speedups, 'ro-', linewidth=2, markersize=6, label='Actual Speedup')
        plt.plot(speedup_processes, speedup_processes, 'k--', alpha=0.5, label='Ideal Speedup')
        plt.xlabel('Number of MPI Processes')
        plt.ylabel('Speedup')
        plt.title('MPI Speedup Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot efficiency
    if "efficiency_analysis" in performance:
        eff_processes = [int(p) for p in performance["efficiency_analysis"].keys()]
        eff_processes.sort()
        efficiencies = [performance["efficiency_analysis"][str(p)] for p in eff_processes]
        
        plt.subplot(2, 2, 3)
        plt.plot(eff_processes, efficiencies, 'go-', linewidth=2, markersize=6)
        plt.xlabel('Number of MPI Processes')
        plt.ylabel('Efficiency')
        plt.title('MPI Efficiency Analysis')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    # Plot throughput (formulas per second)
    plt.subplot(2, 2, 4)
    throughput = [1.0/t if t > 0 else 0 for t in times]
    plt.plot(process_counts, throughput, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Throughput (formulas/second)')
    plt.title('MPI Throughput vs Number of Processes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/MPI_Performance_Scaling_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Performance scaling chart saved to: Results/MPI_Performance_Scaling_Analysis.png")


def create_formula_success_chart(results: Dict[str, Any]) -> None:
    """Create formula success rate chart"""
    
    print("Creating formula success rate chart...")
    
    scaling = results["scaling_analysis"]
    formula_results = results["formula_results"]
    
    # Get success rates for each formula
    formula_ids = []
    success_rates = []
    avg_times = []
    
    for formula_id, formula_data in formula_results.items():
        if "mpi_results" in formula_data and formula_data["mpi_results"]:
            # Calculate average success rate across all process counts
            success_count = 0
            total_count = 0
            times = []
            
            for process_data in formula_data["mpi_results"].values():
                if process_data["solved"]:
                    success_count += 1
                total_count += 1
                times.append(process_data["time"])
            
            if total_count > 0:
                success_rate = success_count / total_count
                avg_time = np.mean(times) if times else 0
                
                formula_ids.append(int(formula_id))
                success_rates.append(success_rate)
                avg_times.append(avg_time)
    
    # Sort by formula ID
    sorted_data = sorted(zip(formula_ids, success_rates, avg_times))
    formula_ids, success_rates, avg_times = zip(*sorted_data)
    
    plt.figure(figsize=(15, 10))
    
    # Plot success rates
    plt.subplot(2, 2, 1)
    colors = ['green' if sr == 1.0 else 'orange' if sr > 0.5 else 'red' for sr in success_rates]
    plt.bar(formula_ids, success_rates, color=colors, alpha=0.7)
    plt.xlabel('Formula ID')
    plt.ylabel('Success Rate')
    plt.title('Formula Success Rate Across All MPI Process Counts')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot average execution times
    plt.subplot(2, 2, 2)
    plt.bar(formula_ids, avg_times, alpha=0.7, color='blue')
    plt.xlabel('Formula ID')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Average Execution Time per Formula')
    plt.grid(True, alpha=0.3)
    
    # Plot success rate distribution
    plt.subplot(2, 2, 3)
    plt.hist(success_rates, bins=10, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Success Rate')
    plt.ylabel('Number of Formulas')
    plt.title('Distribution of Formula Success Rates')
    plt.grid(True, alpha=0.3)
    
    # Plot time distribution
    plt.subplot(2, 2, 4)
    plt.hist(avg_times, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Average Execution Time (seconds)')
    plt.ylabel('Number of Formulas')
    plt.title('Distribution of Average Execution Times')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/MPI_Formula_Success_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Formula success chart saved to: Results/MPI_Formula_Success_Analysis.png")


def create_process_comparison_chart(results: Dict[str, Any]) -> None:
    """Create detailed process comparison chart"""
    
    print("Creating process comparison chart...")
    
    formula_results = results["formula_results"]
    
    # Get all process counts
    all_processes = set()
    for formula_data in formula_results.values():
        if "mpi_results" in formula_data:
            all_processes.update(formula_data["mpi_results"].keys())
    
    process_counts = sorted([int(p) for p in all_processes])
    
    # Collect data for each process count
    process_data = {}
    for process_count in process_counts:
        solved_count = 0
        total_time = 0
        total_formulas = 0
        times = []
        
        for formula_data in formula_results.values():
            if "mpi_results" in formula_data and str(process_count) in formula_data["mpi_results"]:
                result = formula_data["mpi_results"][str(process_count)]
                if result["solved"]:
                    solved_count += 1
                total_time += result["time"]
                times.append(result["time"])
                total_formulas += 1
        
        if total_formulas > 0:
            process_data[process_count] = {
                "success_rate": solved_count / total_formulas,
                "avg_time": total_time / total_formulas,
                "total_solved": solved_count,
                "total_formulas": total_formulas,
                "times": times
            }
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Success rate vs processes
    process_counts = list(process_data.keys())
    success_rates = [process_data[p]["success_rate"] for p in process_counts]
    
    axes[0, 0].plot(process_counts, success_rates, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of MPI Processes')
    axes[0, 0].set_ylabel('Overall Success Rate')
    axes[0, 0].set_title('Overall Success Rate vs Number of Processes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # Average time vs processes
    avg_times = [process_data[p]["avg_time"] for p in process_counts]
    
    axes[0, 1].plot(process_counts, avg_times, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Number of MPI Processes')
    axes[0, 1].set_ylabel('Average Time per Formula (seconds)')
    axes[0, 1].set_title('Average Time vs Number of Processes')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Total solved vs processes
    total_solved = [process_data[p]["total_solved"] for p in process_counts]
    
    axes[1, 0].bar(process_counts, total_solved, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of MPI Processes')
    axes[1, 0].set_ylabel('Total Formulas Solved')
    axes[1, 0].set_title('Total Formulas Solved vs Number of Processes')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time distribution for different process counts
    selected_processes = [1, 8, 16, 32] if len(process_counts) >= 4 else process_counts[:4]
    
    for i, process_count in enumerate(selected_processes):
        if process_count in process_data:
            times = process_data[process_count]["times"]
            axes[1, 1].hist(times, bins=10, alpha=0.5, label=f'{process_count} processes')
    
    axes[1, 1].set_xlabel('Execution Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Time Distribution for Different Process Counts')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/MPI_Process_Comparison_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Process comparison chart saved to: Results/MPI_Process_Comparison_Analysis.png")


def create_detailed_formula_analysis(results: Dict[str, Any]) -> None:
    """Create detailed analysis for specific formulas"""
    
    print("Creating detailed formula analysis...")
    
    formula_results = results["formula_results"]
    
    # Select interesting formulas for detailed analysis
    interesting_formulas = []
    for formula_id, formula_data in formula_results.items():
        if "mpi_results" in formula_data and len(formula_data["mpi_results"]) > 1:
            # Check if results vary across process counts
            times = [result["time"] for result in formula_data["mpi_results"].values()]
            if max(times) > min(times) * 1.1:  # At least 10% variation
                interesting_formulas.append((int(formula_id), formula_data))
    
    # Take first 4 interesting formulas
    interesting_formulas = sorted(interesting_formulas, key=lambda x: x[0])[:4]
    
    if not interesting_formulas:
        print("No formulas with significant variation found for detailed analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (formula_id, formula_data) in enumerate(interesting_formulas):
        if i >= 4:
            break
            
        mpi_results = formula_data["mpi_results"]
        
        # Extract data
        processes = sorted([int(p) for p in mpi_results.keys()])
        times = [mpi_results[str(p)]["time"] for p in processes]
        solved = [mpi_results[str(p)]["solved"] for p in processes]
        
        # Plot time vs processes
        ax = axes[i]
        colors = ['green' if s else 'red' for s in solved]
        ax.scatter(processes, times, c=colors, s=60, alpha=0.7)
        ax.plot(processes, times, 'b-', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Number of MPI Processes')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Formula {formula_id}: Time vs Processes\n'
                    f'Length: {formula_data["len_formula"]} clauses')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Solved'),
                          Patch(facecolor='red', label='Not Solved')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Results/MPI_Detailed_Formula_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Detailed formula analysis saved to: Results/MPI_Detailed_Formula_Analysis.png")


def create_summary_report(results: Dict[str, Any]) -> None:
    """Create a text summary report"""
    
    print("Creating summary report...")
    
    metadata = results["analysis_metadata"]
    performance = results["performance_analysis"]
    
    report = f"""
MPI BRUTE FORCE SOLVER - COMPREHENSIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

ANALYSIS METADATA:
- Total processes tested: {metadata['total_processes_tested']}
- Successful runs: {metadata['successful_runs']}
- Failed runs: {metadata['failed_runs']}
- Success rate: {metadata['successful_runs']/metadata['total_processes_tested']*100:.1f}%
- Total formulas tested: {metadata['total_formulas']}

PERFORMANCE SUMMARY:
"""
    
    if "average_times_per_process" in performance:
        report += "\nAverage execution times per formula:\n"
        for processes, avg_time in sorted(performance["average_times_per_process"].items(), key=lambda x: int(x[0])):
            report += f"  {processes:2s} processes: {avg_time:.6f}s\n"
    
    if "speedup_analysis" in performance:
        report += "\nSpeedup analysis (relative to 1 process):\n"
        for processes, speedup in sorted(performance["speedup_analysis"].items(), key=lambda x: int(x[0])):
            report += f"  {processes:2s} processes: {speedup:.2f}x speedup\n"
    
    if "efficiency_analysis" in performance:
        report += "\nEfficiency analysis:\n"
        for processes, efficiency in sorted(performance["efficiency_analysis"].items(), key=lambda x: int(x[0])):
            report += f"  {processes:2s} processes: {efficiency:.2f} ({efficiency*100:.1f}%)\n"
    
    # Formula success analysis
    formula_results = results["formula_results"]
    total_formulas = len(formula_results)
    solved_formulas = sum(1 for f in formula_results.values() 
                         if "mpi_results" in f and any(r["solved"] for r in f["mpi_results"].values()))
    
    report += f"\nFORMULA ANALYSIS:\n"
    report += f"- Total formulas: {total_formulas}\n"
    report += f"- Formulas that were solved at least once: {solved_formulas}\n"
    report += f"- Overall success rate: {solved_formulas/total_formulas*100:.1f}%\n"
    
    # Save report
    with open('Results/MPI_Analysis_Summary_Report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Summary report saved to: Results/MPI_Analysis_Summary_Report.txt")


def main():
    """Main function to generate all MPI charts and reports"""
    
    print("=" * 80)
    print("MPI BRUTE FORCE SOLVER - CHART GENERATOR")
    print("=" * 80)
    
    # Create Results directory if it doesn't exist
    os.makedirs('Results', exist_ok=True)
    
    # Load results
    print("Loading MPI results...")
    results = load_mpi_results()
    print(f"✓ Loaded results for {results['analysis_metadata']['total_processes_tested']} process configurations")
    
    # Generate charts
    try:
        create_performance_scaling_chart(results)
        create_formula_success_chart(results)
        create_process_comparison_chart(results)
        create_detailed_formula_analysis(results)
        create_summary_report(results)
        
        print("\n" + "=" * 80)
        print("ALL CHARTS AND REPORTS GENERATED SUCCESSFULLY")
        print("=" * 80)
        print("Generated files:")
        print("  - Results/MPI_Performance_Scaling_Analysis.png")
        print("  - Results/MPI_Formula_Success_Analysis.png")
        print("  - Results/MPI_Process_Comparison_Analysis.png")
        print("  - Results/MPI_Detailed_Formula_Analysis.png")
        print("  - Results/MPI_Analysis_Summary_Report.txt")
        
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
