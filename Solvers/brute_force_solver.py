from Solvers.cnf_solver_interface import CNFSolverInterface


class BruteForceSolver(CNFSolverInterface):
    def solve(self):
        print("Hello!", self.variables)
