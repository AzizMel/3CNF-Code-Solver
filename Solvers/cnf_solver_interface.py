from typing import List, Dict


class CNFSolverInterface:
    variables = []
    formula = []

    def __init__(self, variables, formula):
        self.variables = variables
        self.formula = formula

    def solve(self):
        raise NotImplementedError("Please Implement this method")

    def evaluate_clause(self, clause: List[str], assignment: Dict[str, bool]) -> bool:
        for literal in clause:
            if literal.startswith("-"):  # negated literal
                var = literal[1:]
                if var in assignment and not assignment[var]:
                    return True
            else:  # positive literal
                if literal in assignment and assignment[literal]:
                    return True
        return False

    def evaluate_formula(self, assignment: Dict[str, bool]) -> bool:
        return all(self.evaluate_clause(clause, assignment) for clause in self.formula)
