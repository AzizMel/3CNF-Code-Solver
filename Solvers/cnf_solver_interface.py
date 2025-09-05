class CNFSolverInterface:
    variables = []
    formula = []

    def __init__(self, variables, formula):
        self.variables = variables
        self.formula = formula

        def solve(self):
            raise NotImplementedError("Please Implement this method")
