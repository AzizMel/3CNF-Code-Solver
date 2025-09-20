#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import sys
import os

# Add the parent directory to the Python path to find the Solvers module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both current directory and parent directory to path
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

print(f"Current file: {__file__}")
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:5]}")

# Test import
try:
    from Solvers.brute_force_solver import BruteForceSolver
    print("✓ Import successful!")
    print(f"BruteForceSolver class: {BruteForceSolver}")
    
    # Test creating an instance
    test_formula = [["x1", "x2"], ["-x1", "-x2"]]
    solver = BruteForceSolver(test_formula)
    solver.variables = ["x1", "x2"]
    
    result = solver.solve()
    print(f"Test solve result: {result}")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
