# fining best method:
import devTools
import formulas_data
from Solvers.dpll_solver import DPLLSolver
from Solvers.walk_sat_solver import WalkSatSolver
from Solvers.brute_force_solver import BruteForceSolver

formula_to_solve = formulas_data.solvable_formulas[1]

dpll_solver = DPLLSolver(formula_to_solve)
walk_sat_solver = WalkSatSolver(formula_to_solve)
brute_force_solver = BruteForceSolver(formula_to_solve)

brute_force_solver_result = devTools.run_function_with_timing(brute_force_solver.solve)
walk_sat_solver_result = devTools.run_function_with_timing(walk_sat_solver.solve)
dpll_solver_result = devTools.run_function_with_timing(dpll_solver.solve)

print("formula_to_solve()=>", formula_to_solve)
print("len(formula_to_solve)=>", len(formula_to_solve))
print("dpll_solver.solve()=>", dpll_solver_result[0][1])
print("brute_force_solver.solve()=>", brute_force_solver_result[0][1])
print("walk_sat_solver.solve()=>", walk_sat_solver_result[0][1])
