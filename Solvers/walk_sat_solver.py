from Solvers.cnf_solver_interface import CNFSolverInterface
from typing import List, Dict, Optional
import random


class WalkSatSolver(CNFSolverInterface):
    def solve(self, max_flips: int = 10000, p: float = 0.5) -> Optional[Dict[str, bool]]:
        """WalkSAT algorithm - randomized local search"""

        # Random initial assignment
        assignment = {var: random.choice([True, False]) for var in self.variables}

        for _ in range(max_flips):

            # Check if current assignment satisfies formula
            if self.evaluate_formula(assignment):
                return assignment

            # Find unsatisfied clauses
            unsatisfied_clauses = []
            for clause in self.formula:
                if not self.evaluate_clause(clause, assignment):
                    unsatisfied_clauses.append(clause)

            if not unsatisfied_clauses:
                return assignment

            # Pick random unsatisfied clause
            clause = random.choice(unsatisfied_clauses)

            # With probability p, flip random variable in clause
            if random.random() < p:
                literal = random.choice(clause)
                var = literal[1:] if literal.startswith("-") else literal
                assignment[var] = not assignment[var]
            else:
                # Otherwise, flip variable that minimizes number of unsatisfied clauses
                best_var = None
                best_count = float("inf")

                for literal in clause:
                    var = literal[1:] if literal.startswith("-") else literal
                    # Try flipping this variable
                    assignment[var] = not assignment[var]

                    # Count unsatisfied clauses
                    unsatisfied_count = sum(1
                                            for c in self.formula
                                            if not self.evaluate_clause(c, assignment))

                    if unsatisfied_count < best_count:
                        best_count = unsatisfied_count
                        best_var = var

                    # Flip back
                    assignment[var] = not assignment[var]

                if best_var:
                    assignment[best_var] = not assignment[best_var]

        return None
