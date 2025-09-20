#!/usr/bin/env python3
"""
Small test script for MPI brute force solver
Tests with just a few processes to verify everything works
Usage: python test_mpi_small.py
"""

import os
import sys
import subprocess
import json
import time

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def test_mpi_processes(process_counts: list):
    """Test MPI with specified process counts"""
    
    print("=" * 60)
    print("MPI SMALL SCALE TEST")
    print("=" * 60)
    print(f"Testing process counts: {process_counts}")
    
    results = []
    
    for num_processes in process_counts:
        print(f"\nTesting {num_processes} processes...")
        
        cmd = [
            "mpiexec", 
            "-n", str(num_processes),
            sys.executable, 
            "mpi/run_mpi_brute_force.py"
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=parent_dir,
                timeout=60  # 1 minute timeout for test
            )
            
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✓ Success with {num_processes} processes ({end_time - start_time:.2f}s)")
                
                # Load results from Results folder
                results_file = f"Results/mpi_brute_force_results_{num_processes}proc.json"
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    solved_count = sum(1 for r in data.values() 
                                     if r["mpi_brute_force_solver"]["solved"])
                    total_count = len(data)
                    avg_time = sum(r["mpi_brute_force_solver"]["time"] for r in data.values()) / total_count
                    
                    print(f"  Solved: {solved_count}/{total_count} formulas")
                    print(f"  Average time: {avg_time:.6f}s per formula")
                    
                    results.append({
                        "processes": num_processes,
                        "success": True,
                        "solved": solved_count,
                        "total": total_count,
                        "avg_time": avg_time,
                        "total_time": end_time - start_time
                    })
                else:
                    print(f"⚠ Results file not found: {results_file}")
            else:
                print(f"✗ Failed with {num_processes} processes")
                print(f"  Error: {result.stderr}")
                
                results.append({
                    "processes": num_processes,
                    "success": False,
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout with {num_processes} processes")
            results.append({
                "processes": num_processes,
                "success": False,
                "error": "Timeout"
            })
        except Exception as e:
            print(f"✗ Exception with {num_processes} processes: {e}")
            results.append({
                "processes": num_processes,
                "success": False,
                "error": str(e)
            })
    
    return results


def main():
    """Run small scale MPI test"""
    
    # Test with small number of processes first
    test_processes = [1, 2, 4]
    
    print("Starting small scale MPI test...")
    print("This will test the MPI solver with a few processes to verify it works.")
    
    results = test_mpi_processes(test_processes)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Successful tests: {len(successful)}/{len(results)}")
    print(f"Failed tests: {len(failed)}")
    
    if successful:
        print("\nSuccessful results:")
        for r in successful:
            print(f"  {r['processes']} processes: {r['solved']}/{r['total']} solved, "
                  f"avg {r['avg_time']:.6f}s per formula")
    
    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  {r['processes']} processes: {r.get('error', 'Unknown error')}")
    
    # Save test results to Results folder
    os.makedirs('Results', exist_ok=True)
    with open('Results/mpi_small_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: Results/mpi_small_test_results.json")
    
    if len(successful) == len(results):
        print("\n✓ All tests passed! The MPI solver is working correctly.")
        print("You can now run the full scaling analysis with: python run_mpi_scaling_analysis.py")
    else:
        print("\n⚠ Some tests failed. Please check the errors above before running the full analysis.")


if __name__ == "__main__":
    main()
