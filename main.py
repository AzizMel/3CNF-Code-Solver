# fining best method:
import formulas_data
from Solvers.dpll_solver import DPLLSolver
from Solvers.walk_sat_solver import WalkSatSolver
from Solvers.brute_force_solver import BruteForceSolver

formelaToSolve = formulas_data.solvable_formulas[1]
dpll_solver = DPLLSolver(['x1', 'x2', 'x3'], formelaToSolve)
walk_sat_solver = WalkSatSolver(['x1', 'x2', 'x3'], formelaToSolve)
brute_force_solver = BruteForceSolver(['x1', 'x2', 'x3'], formelaToSolve)

print("brute_force_solver.solve()=>",brute_force_solver.solve())
print("walk_sat_solver.solve()=>",walk_sat_solver.solve())
print("dpll_solver.solve()=>",dpll_solver.solve())

