import itertools


class CNFSolver:
    variables = None
    formula = None

    def __init__(self, formula, variables):
        self.formula = formula
        self.variables = variables

    def evaluate_clause(self, clause, assignment):
        for literal in clause:
            if literal.startswith("-"):  # negated literal
                var = literal[1:]
                if not assignment[var]:  # ¬var is True
                    return True
            else:  # positive literal
                if assignment[literal]:
                    return True
        return False  # clause is False if no literal satisfied

    def evaluate_formula(self, assignment):
        return all(self.evaluate_clause(clause, assignment) for clause in self.formula)

    def is_solvable(self):
        for values in itertools.product([False, True], repeat=len(self.variables)):
            assignment = dict(zip(self.variables, values))
            if self.evaluate_formula(assignment=assignment):
                print("Satisfying assignment found:", assignment)
                return assignment
        return None
