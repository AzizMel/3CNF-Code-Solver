from typing import List, Dict, Optional, Set
from Solvers.cnf_solver_interface import CNFSolverInterface


class DPLLSolver(CNFSolverInterface):
    def solve(self) -> Optional[Dict[str, bool]]:
        """DPLL (Davis-Putnam-Logemann-Loveland) algorithm with corrections"""
        return self.dpll_recursive(self.formula, {})

    def dpll_recursive(self, formula: List[List[str]], assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        # Simplify the formula based on current assignment
        simplified_formula, empty_clause_found = self.simplify_formula(formula, assignment)

        # Check for termination conditions
        if empty_clause_found:
            return None
        if not simplified_formula:
            return assignment

        # Unit propagation
        unit_result = self.unit_propagation(simplified_formula, assignment)
        if unit_result is not None:
            return unit_result

        # Pure literal elimination
        pure_result = self.pure_literal_elimination(simplified_formula, assignment)
        if pure_result is not None:
            return pure_result

        # Branching - choose an unassigned variable from the current formula
        current_vars = self.get_variables_from_formula(simplified_formula)
        unassigned_vars = current_vars - set(assignment.keys())

        if not unassigned_vars:
            # This should not happen if simplification is correct
            return None

        var = next(iter(unassigned_vars))

        # Try assigning True to the variable
        true_assignment = assignment.copy()
        true_assignment[var] = True
        result = self.dpll_recursive(simplified_formula, true_assignment)
        if result is not None:
            return result

        # Try assigning False to the variable
        false_assignment = assignment.copy()
        false_assignment[var] = False
        return self.dpll_recursive(simplified_formula, false_assignment)

    def simplify_formula(self, formula: List[List[str]], assignment: Dict[str, bool]) -> (List[List[str]], bool):
        """Simplify formula based on current assignment, return simplified formula and whether an empty clause was found"""
        simplified_formula = []
        empty_clause_found = False

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
                    empty_clause_found = True
                    break
                simplified_formula.append(new_clause)

        return simplified_formula, empty_clause_found

    def unit_propagation(self, formula: List[List[str]], assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        """Find and process all unit clauses"""
        for clause in formula:
            if len(clause) == 1:
                literal = clause[0]
                new_assignment = assignment.copy()

                if literal.startswith("-"):
                    var = literal[1:]
                    if var in assignment and assignment[var]:  # Conflict
                        return None
                    new_assignment[var] = False
                else:
                    if literal in assignment and not assignment[literal]:  # Conflict
                        return None
                    new_assignment[literal] = True

                # Recursively solve with the new assignment
                return self.dpll_recursive(formula, new_assignment)

        return None

    def pure_literal_elimination(self, formula: List[List[str]], assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        """Find and process all pure literals"""
        positive_vars = set()
        negative_vars = set()

        for clause in formula:
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

            # Recursively solve with the new assignment
            return self.dpll_recursive(formula, new_assignment)

        return None

    def get_variables_from_formula(self, formula: List[List[str]]) -> Set[str]:
        """Extract all variables from the current formula"""
        variables = set()
        for clause in formula:
            for literal in clause:
                if literal.startswith("-"):
                    variables.add(literal[1:])
                else:
                    variables.add(literal)
        return variables