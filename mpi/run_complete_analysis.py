#!/usr/bin/env python3
"""
Complete MPI Analysis Runner
Runs the entire MPI brute force analysis pipeline
Usage: python run_complete_analysis.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        end_time = time.time()
        print(f"✓ {description} completed successfully ({end_time - start_time:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False

def main():
    """Run complete MPI analysis pipeline"""
    
    print("=" * 80)
    print("MPI BRUTE FORCE SOLVER - COMPLETE ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Test MPI setup
    print("\nSTEP 1: Testing MPI Setup")
    if not run_command([sys.executable, "mpi/tests/test_mpi_small.py"], "Small-scale MPI test"):
        print("\n❌ MPI setup test failed. Please check your MPI installation.")
        print("Try installing MPI with: sudo apt-get install mpich libmpich-dev")
        return
    
    # Step 2: Run full scaling analysis
    print("\nSTEP 2: Running Full Scaling Analysis")
    if not run_command([sys.executable, "mpi/run_mpi_scaling_analysis.py"], "Full MPI scaling analysis"):
        print("\n❌ Scaling analysis failed. Check the error messages above.")
        return
    
    # Step 3: Generate charts
    print("\nSTEP 3: Generating Charts and Reports")
    if not run_command([sys.executable, "mpi/generate_mpi_charts.py"], "Chart generation"):
        print("\n❌ Chart generation failed. Check the error messages above.")
        return
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nGenerated files:")
    print("📊 Reports:")
    print("  - Results/MPIAnalysis/mpi_comprehensive_scaling_report.json")
    print("  - Results/MPIAnalysis/mpi_raw_scaling_results.json") 
    print("  - Results/MPIAnalysis/mpi_scaling_summary.json")
    print("  - Results/MPIAnalysis/mpi_brute_force_results_Xproc.json")
    
    print("\n📈 Charts:")
    print("  - Results/MPI_Performance_Scaling_Analysis.png")
    print("  - Results/MPI_Formula_Success_Analysis.png")
    print("  - Results/MPI_Process_Comparison_Analysis.png")
    print("  - Results/MPI_Detailed_Formula_Analysis.png")
    print("  - Results/MPI_Analysis_Summary_Report.txt")
    
    print("\n🚀 You can now analyze the results and charts!")

if __name__ == "__main__":
    main()
