# MPI Directory Structure

This document describes the organized structure of the MPI brute force solver implementation.

## Directory Layout

```
CNFCodes/
├── mpi/                                    # MPI Implementation Directory
│   ├── mpi_brute_force_solver.py          # Core MPI solver implementation
│   ├── run_mpi_brute_force.py             # Simple MPI runner
│   ├── run_mpi_scaling_analysis.py        # Comprehensive scaling analysis
│   ├── generate_mpi_charts.py             # Chart generation
│   ├── run_complete_analysis.py           # Complete pipeline runner
│   ├── README_MPI_Usage.md               # Usage documentation
│   ├── DIRECTORY_STRUCTURE.md            # This file
│   └── tests/                             # Test Files Directory
│       ├── test_mpi_small.py             # Small-scale MPI test
│       └── test_mpi_brute_force.py       # Basic MPI test
│
└── Results/                               # Results Directory
    ├── MPIAnalysis/                       # MPI Analysis Results
    │   ├── mpi_comprehensive_scaling_report.json
    │   ├── mpi_raw_scaling_results.json
    │   ├── mpi_scaling_summary.json
    │   ├── mpi_brute_force_results_Xproc.json
    │   └── mpi_small_test_results.json
    │
    ├── ParallesimAlogs/                   # Parallelism Analysis Logs
    │   ├── Execution Time vs Solved Status.png
    │   ├── Solver Execution Times vs Formula Length.png
    │   ├── Solver Performance vs Formula Size.png
    │   ├── Solver Success-Failure Analysis.png
    │   └── Success Rate of Solvers.png
    │
    └── solver_results.json               # Main solver results
```

## File Organization Principles

### 1. **Separation of Concerns**
- **Core Implementation**: Main MPI solver logic
- **Analysis Scripts**: Scaling analysis and chart generation
- **Test Files**: Organized in dedicated `tests/` subdirectory
- **Results**: Organized by analysis type in subdirectories

### 2. **Results Organization**
- **MPIAnalysis/**: All MPI-related results and reports
- **ParallesimAlogs/**: Existing parallelism analysis charts
- **Root Results/**: Main solver results and shared files

### 3. **Test Organization**
- **mpi/tests/**: Dedicated folder for all test files
- **Clear naming**: `test_*` prefix for easy identification
- **Separate concerns**: Different test files for different purposes

## Usage Examples

### Running Tests
```bash
# Small-scale test
python mpi/tests/test_mpi_small.py

# Basic MPI test
python mpi/tests/test_mpi_brute_force.py
```

### Running Analysis
```bash
# Single MPI run
mpiexec -n 4 python mpi/run_mpi_brute_force.py

# Full scaling analysis
python mpi/run_mpi_scaling_analysis.py

# Generate charts
python mpi/generate_mpi_charts.py

# Complete pipeline
python mpi/run_complete_analysis.py
```

### Output Locations
- **Individual Results**: `Results/MPIAnalysis/mpi_brute_force_results_Xproc.json`
- **Comprehensive Report**: `Results/MPIAnalysis/mpi_comprehensive_scaling_report.json`
- **Charts**: `Results/MPI_*.png`
- **Test Results**: `Results/MPIAnalysis/mpi_small_test_results.json`

## Benefits of This Structure

1. **Clear Organization**: Easy to find specific files and results
2. **Separation of Concerns**: Tests, implementation, and results are clearly separated
3. **Consistent Naming**: Predictable file locations and naming conventions
4. **Scalable**: Easy to add new analysis types or test files
5. **Maintainable**: Clear structure makes code maintenance easier
6. **Professional**: Follows standard software development practices
