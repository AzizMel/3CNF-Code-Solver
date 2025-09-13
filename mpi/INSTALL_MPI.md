# MPI Brute Force Solver Installation Guide

## Prerequisites

Before running the MPI brute force solver, you need to install the required dependencies.

## Installation Steps

### 1. Install mpi4py

```bash
# Using pip
pip install mpi4py

# Or using conda
conda install mpi4py

# Or using apt (Ubuntu/Debian)
sudo apt-get install python3-mpi4py

# Or using yum (CentOS/RHEL)
sudo yum install python3-mpi4py
```

### 2. Install MPI (if not already installed)

#### Ubuntu/Debian:

```bash
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

#### CentOS/RHEL:

```bash
sudo yum install openmpi openmpi-devel
```

#### macOS (using Homebrew):

```bash
brew install openmpi
```

### 3. Verify Installation

Test that MPI is working:

```bash
mpiexec --version
```

Test that mpi4py is working:

```bash
python -c "from mpi4py import MPI; print('MPI4py is working!')"
```

## Running the Solver

### Basic Test (without MPI)

```bash
python3 test_mpi_brute_force.py
```

### Simple MPI Run

```bash
# Run with 2 processes
mpiexec -n 2 python run_mpi_brute_force.py

# Run with 4 processes
mpiexec -n 4 python run_mpi_brute_force.py
```


## Troubleshooting

### Common Issues

1. **"mpiexec: command not found"**

   - Install OpenMPI or MPICH
   - Make sure MPI is in your PATH

2. **"ModuleNotFoundError: No module named 'mpi4py'"**

   - Install mpi4py: `pip install mpi4py`
   - Make sure you're using the same Python environment

3. **"Permission denied"**

   - Make sure the script is executable: `chmod +x run_mpi_brute_force.py`

4. **"No such file or directory"**
   - Make sure you're in the correct directory
   - Check that all required files exist

### Performance Tips

1. **Start small**: Test with 2-4 processes first
2. **Monitor resources**: Each process uses CPU and memory
3. **Formula size**: Larger formulas (6+ variables) may take a long time
4. **Process count**: More processes don't always mean better performance

## Example Output

```
[Rank 0] Starting parallel brute force with 4 processes
[Rank 0] Total assignments: 8
[Rank 0] Assignments per process: ~2
[Rank 0] Searching assignments 0 to 1
[Rank 1] Searching assignments 2 to 3
[Rank 2] Searching assignments 4 to 5
[Rank 3] Searching assignments 6 to 7
[Rank 0] Solution found: {'x1': True, 'x2': False, 'x3': True}
[Rank 0] Parallel brute force complete:
[Rank 0]   Total assignments checked: 4
[Rank 0]   Execution time: 0.001234s
[Rank 0]   Solution found: True
```

## Files Created

- `mpi_brute_force_results_<N>proc.json` - Results from MPI runs
