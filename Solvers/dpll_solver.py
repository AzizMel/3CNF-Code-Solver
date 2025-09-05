from Solvers.cnf_solver_interface import CNFSolverInterface


class DPLLSolver(CNFSolverInterface):
    def solve(self):
        print("Hello!", self.variables)
