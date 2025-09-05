from Solvers.cnf_solver_interface import CNFSolverInterface
from typing import List, Dict, Optional


class DPLLSolver(CNFSolverInterface):
    def solve(self) -> Optional[Dict[str, bool]]:
        """DPLL (Davis-Putnam-Logemann-Loveland) algorithm - much more efficient"""

        def dpll_recursive(formula: List[List[str]], assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:

            # Remove satisfied clauses and unsatisfied literals
            simplified_formula = []
            for clause in formula:
                new_clause = []
                clause_satisfied = False

                for literal in clause:
                    if literal.startswith("-"):
                        var = literal[1:]
                        if var in assignment:
                            if not assignment[var]:  # -var is true
                                clause_satisfied = True
                                break
                            # else: -var is false, ignore this literal
                        else:
                            new_clause.append(literal)
                    else:
                        if literal in assignment:
                            if assignment[literal]:  # var is true
                                clause_satisfied = True
                                break
                            # else: var is false, ignore this literal
                        else:
                            new_clause.append(literal)

                if not clause_satisfied:
                    if not new_clause:  # empty clause = unsatisfiable
                        return None
                    simplified_formula.append(new_clause)

            # If no clauses left, formula is satisfied
            if not simplified_formula:
                return assignment

            # Unit propagation: find unit clauses (single literal)
            for clause in simplified_formula:
                if len(clause) == 1:
                    literal = clause[0]
                    if literal.startswith("-"):
                        var = literal[1:]
                        if var not in assignment:
                            new_assignment = assignment.copy()
                            new_assignment[var] = False
                            return dpll_recursive(simplified_formula, new_assignment)
                    else:
                        if literal not in assignment:
                            new_assignment = assignment.copy()
                            new_assignment[literal] = True
                            return dpll_recursive(simplified_formula, new_assignment)

            # Pure literal elimination
            positive_vars = set()
            negative_vars = set()
            for clause in simplified_formula:
                for literal in clause:
                    if literal.startswith("-"):
                        negative_vars.add(literal[1:])
                    else:
                        positive_vars.add(literal)

            pure_positive = positive_vars - negative_vars
            pure_negative = negative_vars - positive_vars

            if pure_positive or pure_negative:
                new_assignment = assignment.copy()
                for var in pure_positive:
                    if var not in assignment:
                        new_assignment[var] = True
                for var in pure_negative:
                    if var not in assignment:
                        new_assignment[var] = False
                return dpll_recursive(simplified_formula, new_assignment)

            # Choose variable for branching (first unassigned variable)
            unassigned_vars = set(self.variables) - set(assignment.keys())
            if not unassigned_vars:
                return assignment if self.evaluate_formula(assignment) else None

            var = next(iter(unassigned_vars))

            # Try var = True
            true_assignment = assignment.copy()
            true_assignment[var] = True
            result = dpll_recursive(simplified_formula, true_assignment)
            if result is not None:
                return result

            # Try var = False
            false_assignment = assignment.copy()
            false_assignment[var] = False
            return dpll_recursive(simplified_formula, false_assignment)

        return dpll_recursive(self.formula, {})
