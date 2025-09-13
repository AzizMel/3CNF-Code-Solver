import json
import os
import devTools
import formulas_data
import finding_best_parallelism_solver_charts
from Solvers.dpll_solver import DPLLSolver
from Solvers.walk_sat_solver import WalkSatSolver
from Solvers.brute_force_solver import BruteForceSolver

results = {}

for index in formulas_data.formulas:
    formula_to_solve = formulas_data.formulas[index]
    dpll_solver = DPLLSolver(formula_to_solve)
    walk_sat_solver = WalkSatSolver(formula_to_solve)
    brute_force_solver = BruteForceSolver(formula_to_solve)

    brute_force_solver_result = devTools.run_function_with_timing(brute_force_solver.solve)
    walk_sat_solver_result = devTools.run_function_with_timing(walk_sat_solver.solve)
    dpll_solver_result = devTools.run_function_with_timing(dpll_solver.solve)

    results[index] = {
        "formula_to_solve": formula_to_solve,
        "len_formula": len(formula_to_solve),

        "dpll_solver": {
            "time": dpll_solver_result[0][1],
            "solved": dpll_solver_result[0][0] is not None,
            "solution": dpll_solver_result[0][0]
        },
        "brute_force_solver": {
            "time": brute_force_solver_result[0][1],
            "solved": brute_force_solver_result[0][0] is not None,
            "solution": brute_force_solver_result[0][0]
        },
        "walk_sat_solver": {
            "time": walk_sat_solver_result[0][1],
            "solved": walk_sat_solver_result[0][0] is not None,
            "solution": walk_sat_solver_result[0][0]
        }
    }

output_dir = os.path.join(os.path.dirname(__file__), "../Results")
output_file = os.path.join(output_dir, "solver_results.json")
os.makedirs(output_dir, exist_ok=True)
print("output_file=>",output_file)
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)


# finding_best_parallelism_solver_charts.plot_execution_times()
# finding_best_parallelism_solver_charts.plot_time_vs_size()
# finding_best_parallelism_solver_charts.plot_success_failure()
finding_best_parallelism_solver_charts.plot_success_rate()
# finding_best_parallelism_solver_charts.plot_time_vs_solved()

