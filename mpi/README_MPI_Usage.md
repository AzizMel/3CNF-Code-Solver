# MPI Brute Force Solver - Usage Guide

This directory contains the MPI-based parallel brute force solver for 3-CNF formulas with comprehensive scaling analysis capabilities.

## Files Overview

### Core MPI Files:
- `mpi_brute_force_solver.py` - Main MPI solver implementation
- `run_mpi_brute_force.py` - Simple runner for single MPI test
- `run_mpi_scaling_analysis.py` - Comprehensive scaling analysis (1-56 processes)
- `generate_mpi_charts.py` - Chart generator for visualization
- `run_complete_analysis.py` - Complete analysis pipeline runner

### Test Files (in `tests/` folder):
- `tests/test_mpi_small.py` - Small-scale test to verify MPI setup
- `tests/test_mpi_brute_force.py` - Basic MPI brute force test

### Documentation:
- `README_MPI_Usage.md` - This usage guide

## Quick Start

### 1. Test MPI Setup (Recommended First Step)

```bash
cd /media/Learn/IKIU/Master/Done/Term\ 3/Algorithim\ Design/Project\ 3\ \(Final\)/CNFCodes
python mpi/tests/test_mpi_small.py
```

This will test with 1, 2, and 4 processes to verify everything works.

### 2. Run Full Scaling Analysis

```bash
python mpi/run_mpi_scaling_analysis.py
```

This will test processes from 1 to 56 and generate comprehensive reports.

### 3. Generate Charts and Visualizations

```bash
python mpi/generate_mpi_charts.py
```

This creates detailed charts from the MPI results.

## Manual MPI Testing

To test with a specific number of processes:

```bash
mpiexec -n 8 python mpi/run_mpi_brute_force.py
```

Replace `8` with your desired number of processes.

## Output Files

### From Scaling Analysis:
- `Results/MPIAnalysis/mpi_comprehensive_scaling_report.json` - Complete analysis results
- `Results/MPIAnalysis/mpi_raw_scaling_results.json` - Raw test data
- `Results/MPIAnalysis/mpi_scaling_summary.json` - Performance summary
- `Results/MPIAnalysis/mpi_brute_force_results_Xproc.json` - Individual process results

### From Chart Generation:
- `Results/MPI_Performance_Scaling_Analysis.png` - Performance vs processes
- `Results/MPI_Formula_Success_Analysis.png` - Formula success rates
- `Results/MPI_Process_Comparison_Analysis.png` - Process comparison
- `Results/MPI_Detailed_Formula_Analysis.png` - Detailed formula analysis
- `Results/MPI_Analysis_Summary_Report.txt` - Text summary

## Report Format

The generated reports follow the same format as `solver_results.json` but include MPI-specific data:

```json
{
  "0": {
    "formula_to_solve": [...],
    "len_formula": 3,
    "mpi_brute_force_solver": {
      "time": 0.000123,
      "solved": true,
      "solution": {...},
      "assignments_checked": 8,
      "processes": 4,
      "total_assignments": 8
    }
  }
}
```

## Performance Analysis

The comprehensive report includes:

- **Execution Time Analysis**: Time per formula vs number of processes
- **Speedup Analysis**: Actual vs ideal speedup
- **Efficiency Analysis**: Parallel efficiency metrics
- **Success Rate Analysis**: Formula solving success across process counts
- **Scaling Analysis**: How performance scales with process count

## Troubleshooting

### Common Issues:

1. **MPI not found**: Install MPI development tools
   ```bash
   sudo apt-get install mpich libmpich-dev
   ```

2. **Permission denied**: Make scripts executable
   ```bash
   chmod +x mpi/*.py
   ```

3. **Import errors**: Ensure you're running from the project root directory

4. **Timeout errors**: Increase timeout in scripts for larger process counts

### System Requirements:

- Python 3.6+
- MPI implementation (MPICH or OpenMPI)
- mpi4py package
- matplotlib, numpy, pandas for chart generation
- Sufficient memory for parallel processes

## Process Count Recommendations

- **1-8 processes**: Good for development and testing
- **8-16 processes**: Typical for single-node parallelization
- **16-32 processes**: Multi-core systems
- **32-56 processes**: High-core systems or multi-node clusters

Note: Performance may not scale linearly due to communication overhead and problem size limitations.
