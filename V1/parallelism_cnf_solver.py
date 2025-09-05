import itertools
import copy
from typing import List, Dict, Optional, Set


class CNFSolver:
    """Enhanced CNF Solver with multiple algorithms for 3-SAT problems"""

    def __init__(self, formula: List[List[str]], variables: List[str]):
        self.formula = formula
        self.variables = variables
        self.assignments_checked = 0
        self.algorithm_used = ""

    def evaluate_clause(self, clause: List[str], assignment: Dict[str, bool]) -> bool:
        """Evaluate a single clause given variable assignment"""
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
        """Evaluate entire formula given variable assignment"""
        return all(self.evaluate_clause(clause, assignment) for clause in self.formula)

    def brute_force_solve(self) -> Optional[Dict[str, bool]]:
        """Original brute force approach - O(2^n)"""
        self.algorithm_used = "Brute Force"
        self.assignments_checked = 0

        for values in itertools.product([False, True], repeat=len(self.variables)):
            self.assignments_checked += 1
            assignment = dict(zip(self.variables, values))
            if self.evaluate_formula(assignment):
                return assignment
        return None

    def dpll_solve(self) -> Optional[Dict[str, bool]]:
        """DPLL (Davis-Putnam-Logemann-Loveland) algorithm - much more efficient"""
        self.algorithm_used = "DPLL"
        self.assignments_checked = 0

        def dpll_recursive(
            formula: List[List[str]], assignment: Dict[str, bool]
        ) -> Optional[Dict[str, bool]]:
            self.assignments_checked += 1

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

    def walksat_solve(
        self, max_flips: int = 10000, p: float = 0.5
    ) -> Optional[Dict[str, bool]]:
        """WalkSAT algorithm - randomized local search"""
        import random

        self.algorithm_used = "WalkSAT"
        self.assignments_checked = 0

        # Random initial assignment
        assignment = {var: random.choice([True, False]) for var in self.variables}

        for _ in range(max_flips):
            self.assignments_checked += 1

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
                    unsatisfied_count = sum(
                        1
                        for c in self.formula
                        if not self.evaluate_clause(c, assignment)
                    )

                    if unsatisfied_count < best_count:
                        best_count = unsatisfied_count
                        best_var = var

                    # Flip back
                    assignment[var] = not assignment[var]

                if best_var:
                    assignment[best_var] = not assignment[best_var]

        return None

    def solve(self, algorithm: str = "dpll") -> Optional[Dict[str, bool]]:
        """Main solving method with algorithm selection"""
        if algorithm.lower() == "brute_force":
            return self.brute_force_solve()
        elif algorithm.lower() == "dpll":
            return self.dpll_solve()
        elif algorithm.lower() == "walksat":
            return self.walksat_solve()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def is_solvable(self, algorithm: str = "dpll") -> Optional[Dict[str, bool]]:
        """Legacy method for backward compatibility"""
        result = self.solve(algorithm)
        if result:
            print(f"Satisfying assignment found using {self.algorithm_used}:", result)
            print(f"Assignments checked: {self.assignments_checked}")
        return result
