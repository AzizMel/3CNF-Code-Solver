# fining best method:

from Solvers.dpll_solver import DPLLSolver
from Solvers.walk_sat_solver import WalkSatSolver
from Solvers.brute_force_solver import BruteForceSolver

dpll_solver = DPLLSolver(['x1', 'x2', 'x3'], ['hello'])
walk_sat_solver = WalkSatSolver(['x1', 'x2', 'x3'], ['hello'])
brute_force_solver = BruteForceSolver(['x1', 'x2', 'x3'], ['hello'])

dpll_solver.solve()
walk_sat_solver.solve()
brute_force_solver.solve()
