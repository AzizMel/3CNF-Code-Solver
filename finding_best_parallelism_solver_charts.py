import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("solver_results.json", "r") as f:
    results = json.load(f)

formulas = list(results.keys())

# Extract data
dpll_times = [results[i]["dpll_solver"]["time"] for i in formulas]
brute_times = [results[i]["brute_force_solver"]["time"] for i in formulas]
walksat_times = [results[i]["walk_sat_solver"]["time"] for i in formulas]

dpll_solved = [results[i]["dpll_solver"]["solved"] for i in formulas]
brute_solved = [results[i]["brute_force_solver"]["solved"] for i in formulas]
walksat_solved = [results[i]["walk_sat_solver"]["solved"] for i in formulas]

formula_sizes = [results[i]["len_formula"] for i in formulas]

colors = {
    "DPLL": "#1f77b4",        # blue
    "Brute Force": "#ff7f0e", # orange
    "WalkSAT": "#2ca02c"      # green
}


def plot_execution_times():
    """
    Bar chart showing execution times for all solvers,
    grouped by formula length (number of clauses).
    Each formula length will have 3 bars: DPLL, Brute Force, WalkSAT.
    """
    x = np.arange(len(formulas))  # One group per formula
    width = 0.25                  # Width of each bar

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, dpll_times, width, label="DPLL", color=colors["DPLL"], alpha=0.85)
    plt.bar(x, brute_times, width, label="Brute Force", color=colors["Brute Force"], alpha=0.85)
    plt.bar(x + width, walksat_times, width, label="WalkSAT", color=colors["WalkSAT"], alpha=0.85)

    # X-axis labels = formula lengths
    plt.xticks(x, formula_sizes)

    plt.xlabel("Formula Length (# of Clauses)")
    plt.ylabel("Execution Time (s)")
    plt.title("Solver Execution Times vs Formula Length")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_vs_size():
    """
    Line chart showing how solver runtime grows with formula size (# of clauses).
    Each solver is a line, making it easy to compare scaling behavior.
    """
    plt.figure(figsize=(10,5))
    plt.plot(formula_sizes, dpll_times, marker="o", color=colors["DPLL"], label="DPLL")
    plt.plot(formula_sizes, brute_times, marker="s", color=colors["Brute Force"], label="Brute Force")
    plt.plot(formula_sizes, walksat_times, marker="^", color=colors["WalkSAT"], label="WalkSAT")

    plt.xlabel("Number of Clauses in Formula")
    plt.ylabel("Execution Time (s)")
    plt.title("Solver Time vs Formula Size")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_success_failure():
    """
    Stacked bar chart showing how many formulas each solver solved vs failed.
    Green = solved, red = unsolved.
    Good for comparing solver reliability.
    """
    solvers = ["DPLL", "Brute Force", "WalkSAT"]
    solved_counts = [
        sum(dpll_solved),
        sum(brute_solved),
        sum(walksat_solved)
    ]
    unsolved_counts = [
        len(formulas) - sum(dpll_solved),
        len(formulas) - sum(brute_solved),
        len(formulas) - sum(walksat_solved)
    ]

    plt.figure(figsize=(8,6))
    plt.bar(solvers, solved_counts, label="Solved", color="#2ca02c", alpha=0.85)
    plt.bar(solvers, unsolved_counts, bottom=solved_counts, label="Unsolved", color="#d62728", alpha=0.85)

    plt.ylabel("Number of Formulas")
    plt.title("Solver Success/Failure Counts")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_success_rate():
    """
    Pie chart showing success rate (percentage of formulas solved) for each solver.
    Quick way to see overall solver effectiveness.
    """
    solvers = ["DPLL", "Brute Force", "WalkSAT"]
    success_rate = [
        sum(dpll_solved) / len(formulas),
        sum(brute_solved) / len(formulas),
        sum(walksat_solved) / len(formulas)
    ]

    plt.figure(figsize=(8,8))
    plt.pie(success_rate, labels=solvers,
            autopct='%1.1f%%', startangle=140,
            colors=[colors["DPLL"], colors["Brute Force"], colors["WalkSAT"]])
    plt.title("Success Rate of Solvers")
    plt.show()


def plot_time_vs_solved():
    """
    Scatter plot showing execution time vs solved status.
    Each point is a formula.
    Helps visualize whether faster runs tend to succeed or fail.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(dpll_times, dpll_solved, label="DPLL", color=colors["DPLL"], marker="o")
    plt.scatter(brute_times, brute_solved, label="Brute Force", color=colors["Brute Force"], marker="x")
    plt.scatter(walksat_times, walksat_solved, label="WalkSAT", color=colors["WalkSAT"], marker="^")

    plt.xlabel("Execution Time (s)")
    plt.ylabel("Solved (1=True, 0=False)")
    plt.title("Time vs Solved Status")
    plt.legend()
    plt.tight_layout()
    plt.show()
