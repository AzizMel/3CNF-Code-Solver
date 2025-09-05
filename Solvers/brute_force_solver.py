from typing import Dict, Optional
from Solvers.cnf_solver_interface import CNFSolverInterface
import itertools


class BruteForceSolver(CNFSolverInterface):
    def solve(self) -> Optional[Dict[str, bool]]:
        for values in itertools.product([False, True], repeat=len(self.variables)):
            assignment = dict(zip(self.variables, values))
            if self.evaluate_formula(assignment):
                return assignment
        return None
