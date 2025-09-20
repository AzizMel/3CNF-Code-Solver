#!/usr/bin/env python3
"""
Comprehensive MPI scaling analysis script
Tests MPI brute force solver with different numbers of processes (1 to 56)
Generates detailed reports and performance analysis
Usage: python run_mpi_scaling_analysis.py
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import formulas_data


def run_mpi_test(num_processes: int) -> Dict[str, Any]:
    """Run MPI test with specified number of processes"""
    print(f"\n{'='*60}")
    print(f"Testing with {num_processes} MPI processes")
    print(f"{'='*60}")
    
    # Create the mpiexec command
    cmd = [
        "mpiexec", 
        "-n", str(num_processes),
        sys.executable, 
        "mpi/run_mpi_brute_force.py"
    ]
    
    start_time = time.time()
    
    try:
        # Run the MPI command
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=parent_dir,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Successfully completed with {num_processes} processes")
            print(f"  Total execution time: {end_time - start_time:.2f}s")
            
            # Load the results file from Results/MPIAnalysis folder
            results_file = f"Results/MPIAnalysis/mpi_brute_force_results_{num_processes}proc.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                return {
                    "processes": num_processes,
                    "success": True,
                    "execution_time": end_time - start_time,
                    "results": results,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"⚠ Warning: Results file {results_file} not found")
                return {
                    "processes": num_processes,
                    "success": False,
                    "execution_time": end_time - start_time,
                    "error": f"Results file not found: {results_file}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            print(f"✗ Failed with {num_processes} processes")
            print(f"  Return code: {result.returncode}")
            print(f"  Error: {result.stderr}")
            
            return {
                "processes": num_processes,
                "success": False,
                "execution_time": end_time - start_time,
                "error": f"Process failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout with {num_processes} processes")
        return {
            "processes": num_processes,
            "success": False,
            "execution_time": 300,
            "error": "Process timed out after 5 minutes",
            "stdout": "",
            "stderr": "Timeout"
        }
    except Exception as e:
        print(f"✗ Exception with {num_processes} processes: {e}")
        return {
            "processes": num_processes,
            "success": False,
            "execution_time": 0,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }


def generate_comprehensive_report(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive report combining all MPI test results"""
    
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")
    
    # Initialize comprehensive report structure
    comprehensive_report = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_processes_tested": len(all_results),
            "successful_runs": len([r for r in all_results if r["success"]]),
            "failed_runs": len([r for r in all_results if not r["success"]]),
            "total_formulas": len(formulas_data.formulas)
        },
        "formula_results": {},
        "performance_analysis": {},
        "scaling_analysis": {}
    }
    
    # Initialize formula results structure
    for formula_id in formulas_data.formulas.keys():
        comprehensive_report["formula_results"][str(formula_id)] = {
            "formula_to_solve": formulas_data.formulas[formula_id],
            "len_formula": len(formulas_data.formulas[formula_id]),
            "mpi_results": {}
        }
    
    # Process results from each MPI run
    for result in all_results:
        if result["success"]:
            processes = result["processes"]
            formula_results = result["results"]
            
            # Add results for each formula
            for formula_id, formula_data in formula_results.items():
                if formula_id in comprehensive_report["formula_results"]:
                    comprehensive_report["formula_results"][formula_id]["mpi_results"][str(processes)] = {
                        "time": formula_data["mpi_brute_force_solver"]["time"],  # Python will use scientific notation automatically
                        "solved": formula_data["mpi_brute_force_solver"]["solved"],
                        "solution": formula_data["mpi_brute_force_solver"]["solution"],
                        "assignments_checked": formula_data["mpi_brute_force_solver"]["assignments_checked"],
                        "total_assignments": formula_data["mpi_brute_force_solver"]["total_assignments"]
                    }
    
    # Generate performance analysis
    successful_results = [r for r in all_results if r["success"]]
    if successful_results:
        comprehensive_report["performance_analysis"] = {
            "execution_times": {str(r["processes"]): r["execution_time"] for r in successful_results},
            "average_times_per_process": {},
            "speedup_analysis": {},
            "efficiency_analysis": {}
        }
        
        # Calculate average times per process
        for result in successful_results:
            processes = result["processes"]
            total_time = result["execution_time"]
            avg_time = total_time / len(formulas_data.formulas) if len(formulas_data.formulas) > 0 else 0
            comprehensive_report["performance_analysis"]["average_times_per_process"][str(processes)] = avg_time
        
        # Calculate speedup (relative to 1 process)
        if "1" in comprehensive_report["performance_analysis"]["average_times_per_process"]:
            baseline_time = comprehensive_report["performance_analysis"]["average_times_per_process"]["1"]
            for processes, avg_time in comprehensive_report["performance_analysis"]["average_times_per_process"].items():
                if int(processes) > 1:
                    speedup = baseline_time / avg_time if avg_time > 0 else 0
                    efficiency = speedup / int(processes) if int(processes) > 0 else 0
                    
                    comprehensive_report["performance_analysis"]["speedup_analysis"][processes] = speedup
                    comprehensive_report["performance_analysis"]["efficiency_analysis"][processes] = efficiency
    
    # Generate scaling analysis
    comprehensive_report["scaling_analysis"] = {
        "process_counts": [r["processes"] for r in successful_results],
        "success_rates": {},
        "solved_formulas_per_process": {}
    }
    
    # Calculate success rates for each formula across all process counts
    for formula_id in formulas_data.formulas.keys():
        formula_id_str = str(formula_id)
        if formula_id_str in comprehensive_report["formula_results"]:
            solved_counts = []
            for processes in comprehensive_report["scaling_analysis"]["process_counts"]:
                processes_str = str(processes)
                if processes_str in comprehensive_report["formula_results"][formula_id_str]["mpi_results"]:
                    solved = comprehensive_report["formula_results"][formula_id_str]["mpi_results"][processes_str]["solved"]
                    solved_counts.append(1 if solved else 0)
            
            if solved_counts:
                success_rate = sum(solved_counts) / len(solved_counts)
                comprehensive_report["scaling_analysis"]["success_rates"][formula_id_str] = success_rate
    
    return comprehensive_report


def save_reports(comprehensive_report: Dict[str, Any], all_results: List[Dict[str, Any]]) -> None:
    """Save all reports to files in Results/MPIAnalysis folder"""
    
    print(f"\n{'='*60}")
    print("SAVING REPORTS")
    print(f"{'='*60}")
    
    # Create Results/MPIAnalysis directory if it doesn't exist
    os.makedirs("Results/MPIAnalysis", exist_ok=True)
    
    # Save comprehensive report
    comprehensive_file = "Results/MPIAnalysis/mpi_comprehensive_scaling_report.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    print(f"✓ Comprehensive report saved to: {comprehensive_file}")
    
    # Save raw results
    raw_results_file = "Results/MPIAnalysis/mpi_raw_scaling_results.json"
    with open(raw_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Raw results saved to: {raw_results_file}")
    
    # Save individual process results summary
    summary_file = "Results/MPIAnalysis/mpi_scaling_summary.json"
    summary = {
        "metadata": comprehensive_report["analysis_metadata"],
        "performance_summary": comprehensive_report["performance_analysis"],
        "scaling_summary": comprehensive_report["scaling_analysis"]
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary report saved to: {summary_file}")


def print_final_summary(comprehensive_report: Dict[str, Any]) -> None:
    """Print final summary of the analysis"""
    
    print(f"\n{'='*80}")
    print("FINAL SCALING ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    metadata = comprehensive_report["analysis_metadata"]
    print(f"Total processes tested: {metadata['total_processes_tested']}")
    print(f"Successful runs: {metadata['successful_runs']}")
    print(f"Failed runs: {metadata['failed_runs']}")
    print(f"Success rate: {metadata['successful_runs']/metadata['total_processes_tested']*100:.1f}%")
    
    if "performance_analysis" in comprehensive_report and "average_times_per_process" in comprehensive_report["performance_analysis"]:
        print(f"\nPerformance Analysis:")
        print("-" * 40)
        for processes, avg_time in comprehensive_report["performance_analysis"]["average_times_per_process"].items():
            print(f"  {processes:2s} processes: {avg_time:.6f}s average per formula")
        
        if "speedup_analysis" in comprehensive_report["performance_analysis"]:
            print(f"\nSpeedup Analysis (relative to 1 process):")
            print("-" * 40)
            for processes, speedup in comprehensive_report["performance_analysis"]["speedup_analysis"].items():
                print(f"  {processes:2s} processes: {speedup:.2f}x speedup")
        
        if "efficiency_analysis" in comprehensive_report["performance_analysis"]:
            print(f"\nEfficiency Analysis:")
            print("-" * 40)
            for processes, efficiency in comprehensive_report["performance_analysis"]["efficiency_analysis"].items():
                print(f"  {processes:2s} processes: {efficiency:.2f} efficiency ({efficiency*100:.1f}%)")


def main():
    """Main function to run comprehensive MPI scaling analysis"""
    
    print("=" * 80)
    print("MPI BRUTE FORCE SOLVER - COMPREHENSIVE SCALING ANALYSIS")
    print("=" * 80)
    print(f"Testing processes from 1 to 56")
    print(f"Total formulas to test: {len(formulas_data.formulas)}")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Determine process counts to test
    # Test key process counts for comprehensive analysis
    process_counts = [
        1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56
    ]
    
    print(f"Will test {len(process_counts)} different process counts: {process_counts}")
    
    # Run tests
    all_results = []
    start_time = time.time()
    
    for i, num_processes in enumerate(process_counts, 1):
        print(f"\n[{i}/{len(process_counts)}] Testing {num_processes} processes...")
        result = run_mpi_test(num_processes)
        all_results.append(result)
        
        # Small delay between tests
        time.sleep(1)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total analysis time: {total_time:.2f}s")
    print(f"Average time per test: {total_time/len(process_counts):.2f}s")
    
    # Generate comprehensive report
    comprehensive_report = generate_comprehensive_report(all_results)
    
    # Save all reports
    save_reports(comprehensive_report, all_results)
    
    # Print final summary
    print_final_summary(comprehensive_report)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Files generated:")
    print("  - Results/MPIAnalysis/mpi_comprehensive_scaling_report.json (detailed results)")
    print("  - Results/MPIAnalysis/mpi_raw_scaling_results.json (raw test results)")
    print("  - Results/MPIAnalysis/mpi_scaling_summary.json (performance summary)")
    print("  - Results/MPIAnalysis/mpi_brute_force_results_Xproc.json (individual process results)")


if __name__ == "__main__":
    main()
