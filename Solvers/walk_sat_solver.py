from Solvers.cnf_solver_interface import CNFSolverInterface


class WalkSatSolver(CNFSolverInterface):
    def solve(self):
        print("Hello!", self.variables)
