import itertools
import random

import formulas_data


def generate_unsat_3cnf(n_vars=3):
    """
    Generate a canonical unsatisfiable 3-CNF:
    All 2^n assignments blocked by 2^n clauses.
    """
    clauses = []
    for assignment in itertools.product([False, True], repeat=n_vars):
        clause = []
        for i, val in enumerate(assignment, start=1):
            clause.append(f"-x{i}" if val else f"x{i}")
        clauses.append(clause)
    return clauses


def transform_formula(base, n_vars=3):
    """
    Apply a random variable permutation and sign flip to a formula.
    """
    perm = list(range(1, n_vars + 1))
    random.shuffle(perm)

    flips = [random.choice([1, -1]) for _ in range(n_vars)]

    new_formula = []
    for clause in base:
        new_clause = []
        for lit in clause:
            sign = -1 if lit.startswith("-") else 1
            idx = int(lit.lstrip("-x"))
            mapped_idx = perm[idx - 1]
            mapped_sign = sign * flips[idx - 1]
            new_clause.append(("x" if mapped_sign > 0 else "-x") + str(mapped_idx))
        new_formula.append(new_clause)
    return new_formula


def formula_to_set(formula):
    """Convert formula (list of clauses) to a frozenset of frozensets for comparison."""
    return frozenset(frozenset(clause) for clause in formula)


def generate_unsat_batch(existing_formulas, n_new=5, n_vars=3, max_attempts=50000):
    """
    Generate a batch of new unsatisfiable formulas, each different from existing ones.

    existing_formulas: dict or list of formulas
    n_new: number of new formulas to generate
    n_vars: number of variables
    """
    base = generate_unsat_3cnf(n_vars)

    # Normalize existing formulas
    existing_sets = set()
    for f in existing_formulas.values() if isinstance(existing_formulas, dict) else existing_formulas:
        existing_sets.add(formula_to_set(f))

    new_formulas = []
    attempts = 0
    while len(new_formulas) < n_new and attempts < max_attempts:
        attempts += 1
        candidate = transform_formula(base, n_vars)
        cand_set = formula_to_set(candidate)

        if cand_set not in existing_sets:
            new_formulas.append(candidate)
            existing_sets.add(cand_set)

    if len(new_formulas) < n_new:
        print(f"⚠️ Warning: Only generated {len(new_formulas)} unique formulas.")

    return new_formulas


generate_unsat_batch(formulas_data.unsolvable_formulas)