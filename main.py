import ParallelismBruteForceCNFSolver
import devTools
import formulas_data
runtimes = []
results = []


def run_parallelism_brute_force_cnf_solver(formula):
    parallelism_brute_force_cnf_solver = ParallelismBruteForceCNFSolver.CNFSolver(
        variables=['x1', 'x2', 'x3'],
        formula=formula)
    return parallelism_brute_force_cnf_solver.is_solvable()


for formula_id,unsolvable_formula in formulas_data.unsolvable_formulas.items():
    result_time = devTools.run_function_with_timing(run_parallelism_brute_force_cnf_solver,unsolvable_formula)
    results.append(result_time[0][0])
    runtimes.append(result_time[0][1])

print("runtimes =>",runtimes)
print("results =>",results)

