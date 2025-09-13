import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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

# Set up Seaborn styling
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

colors = {
    "DPLL": "#2E86AB",  # professional blue
    "Brute Force": "#F24236",  # professional red
    "WalkSAT": "#43AA8B",  # professional green
}


def plot_execution_times():
    """
    Bar chart showing execution times for all solvers,
    grouped by formula length (number of clauses).
    Each formula length will have 3 bars: DPLL, Brute Force, WalkSAT.
    """
    # Group data by formula length and calculate averages
    unique_lengths = sorted(set(formula_sizes))

    # Calculate average execution times for each solver at each formula length
    dpll_avg_times = []
    brute_avg_times = []
    walksat_avg_times = []

    for length in unique_lengths:
        # Find indices where formula length equals current length
        indices = [i for i, l in enumerate(formula_sizes) if l == length]

        # Calculate average times for this formula length
        dpll_avg = np.mean([dpll_times[i] for i in indices])
        brute_avg = np.mean([brute_times[i] for i in indices])
        walksat_avg = np.mean([walksat_times[i] for i in indices])

        dpll_avg_times.append(dpll_avg)
        brute_avg_times.append(brute_avg)
        walksat_avg_times.append(walksat_avg)

    # Prepare data for Seaborn
    data = []
    for i, length in enumerate(unique_lengths):
        data.append(
            {
                "Formula Length": length,
                "Solver": "DPLL",
                "Average Execution Time (s)": dpll_avg_times[i],
            }
        )
        data.append(
            {
                "Formula Length": length,
                "Solver": "Brute Force",
                "Average Execution Time (s)": brute_avg_times[i],
            }
        )
        data.append(
            {
                "Formula Length": length,
                "Solver": "WalkSAT",
                "Average Execution Time (s)": walksat_avg_times[i],
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=df,
        x="Formula Length",
        y="Average Execution Time (s)",
        hue="Solver",
        palette=colors,
        errorbar=None,  # No confidence interval since we're showing averages
        capsize=0.1,
        err_kws={"linewidth": 2},
    )

    # Customize the plot
    ax.set_title("Solver Execution Times vs Formula Length", fontweight="bold", pad=20)
    ax.set_xlabel("Formula Length (# of Clauses)", fontweight="bold")
    ax.set_ylabel("Average Execution Time (s)", fontweight="bold")

    # Use log scale for y-axis to handle large differences in execution times
    ax.set_yscale("log")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2e", fontsize=10, rotation=90)

    # Improve legend
    ax.legend(
        title="Solver",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


def plot_time_vs_size():
    """
    Line chart showing how solver runtime grows with formula size (# of clauses).
    Each solver is a line, making it easy to compare scaling behavior.
    """
    # Prepare data for Seaborn
    data = []
    for i, length in enumerate(formula_sizes):
        data.append(
            {
                "Formula Length": length,
                "Solver": "DPLL",
                "Execution Time (s)": dpll_times[i],
            }
        )
        data.append(
            {
                "Formula Length": length,
                "Solver": "Brute Force",
                "Execution Time (s)": brute_times[i],
            }
        )
        data.append(
            {
                "Formula Length": length,
                "Solver": "WalkSAT",
                "Execution Time (s)": walksat_times[i],
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df,
        x="Formula Length",
        y="Execution Time (s)",
        hue="Solver",
        style="Solver",
        markers=True,
        markersize=8,
        linewidth=3,
        palette=colors,
    )

    # Customize the plot
    ax.set_title("Solver Performance vs Formula Size", fontweight="bold", pad=20)
    ax.set_xlabel("Number of Clauses in Formula", fontweight="bold")
    ax.set_ylabel("Execution Time (s)", fontweight="bold")

    # Improve legend
    ax.legend(
        title="Solver",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set y-axis to log scale for better visualization of time differences
    ax.set_yscale("log")

    plt.tight_layout()
    plt.show()


def plot_success_failure():
    """
    Stacked bar chart showing how many formulas each solver solved vs failed.
    Green = solved, red = unsolved.
    Good for comparing solver reliability.
    """
    # Prepare data for Seaborn
    solvers = ["DPLL", "Brute Force", "WalkSAT"]
    solved_counts = [sum(dpll_solved), sum(brute_solved), sum(walksat_solved)]
    unsolved_counts = [
        len(formulas) - sum(dpll_solved),
        len(formulas) - sum(brute_solved),
        len(formulas) - sum(walksat_solved),
    ]

    data = []
    for solver, solved, unsolved in zip(solvers, solved_counts, unsolved_counts):
        data.append({"Solver": solver, "Count": solved, "Status": "Solved"})
        data.append({"Solver": solver, "Count": unsolved, "Status": "Unsolved"})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=df,
        x="Solver",
        y="Count",
        hue="Status",
        palette=["#43AA8B", "#F24236"],  # Green for solved, Red for unsolved
        alpha=0.85,
    )

    # Customize the plot
    ax.set_title("Solver Success/Failure Analysis", fontweight="bold", pad=20)
    ax.set_xlabel("Solver", fontweight="bold")
    ax.set_ylabel("Number of Formulas", fontweight="bold")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=12, fontweight="bold")

    # Improve legend
    ax.legend(
        title="Status",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.show()


def plot_success_rate():
    """
    Donut chart showing success rate (percentage of formulas solved) for each solver.
    Quick way to see overall solver effectiveness.
    """
    solvers = ["DPLL", "Brute Force", "WalkSAT"]
    success_rate = [
        sum(dpll_solved) / len(formulas),
        sum(brute_solved) / len(formulas),
        sum(walksat_solved) / len(formulas),
    ]

    # Calculate failure rates
    failure_rate = [1 - rate for rate in success_rate]

    # Create subplots for success and failure rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Success rate donut chart
    wedges, texts, autotexts = ax1.pie(
        success_rate,
        labels=solvers,
        autopct="%1.1f%%",
        startangle=90,
        colors=[colors["DPLL"], colors["Brute Force"], colors["WalkSAT"]],
        pctdistance=0.85,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    # Create donut by adding a white circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax1.add_artist(centre_circle)
    ax1.set_title("Success Rate of Solvers", fontsize=16, fontweight="bold", pad=20)

    # Add percentage text in the center
    ax1.text(
        0,
        0,
        f"Total\n{len(formulas)} Formulas",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Failure rate donut chart
    wedges2, texts2, autotexts2 = ax2.pie(
        failure_rate,
        labels=solvers,
        autopct="%1.1f%%",
        startangle=90,
        colors=[colors["DPLL"], colors["Brute Force"], colors["WalkSAT"]],
        pctdistance=0.85,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    # Create donut by adding a white circle in the center
    centre_circle2 = plt.Circle((0, 0), 0.70, fc="white")
    ax2.add_artist(centre_circle2)
    ax2.set_title("Failure Rate of Solvers", fontsize=16, fontweight="bold", pad=20)

    # Add percentage text in the center
    ax2.text(
        0,
        0,
        f"Total\n{len(formulas)} Formulas",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Make axes equal
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def plot_time_vs_solved():
    """
    Scatter plot showing execution time vs solved status.
    Each point is a formula.
    Helps visualize whether faster runs tend to succeed or fail.
    """
    # Prepare data for Seaborn
    data = []
    for i in range(len(formulas)):
        data.append(
            {
                "Execution Time (s)": dpll_times[i],
                "Solved": "Yes" if dpll_solved[i] else "No",
                "Solver": "DPLL",
            }
        )
        data.append(
            {
                "Execution Time (s)": brute_times[i],
                "Solved": "Yes" if brute_solved[i] else "No",
                "Solver": "Brute Force",
            }
        )
        data.append(
            {
                "Execution Time (s)": walksat_times[i],
                "Solved": "Yes" if walksat_solved[i] else "No",
                "Solver": "WalkSAT",
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 8))
    ax = sns.scatterplot(
        data=df,
        x="Execution Time (s)",
        y="Solved",
        hue="Solver",
        style="Solver",
        s=100,
        alpha=0.7,
        palette=colors,
    )

    # Customize the plot
    ax.set_title("Execution Time vs Solved Status", fontweight="bold", pad=20)
    ax.set_xlabel("Execution Time (s)", fontweight="bold")
    ax.set_ylabel("Solved Status", fontweight="bold")

    # Set x-axis to log scale for better visualization
    ax.set_xscale("log")

    # Improve legend
    ax.legend(
        title="Solver",
        title_fontsize=14,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to generate all charts with improved UI using Seaborn.
    """
    print("Generating improved charts with Seaborn styling...")

    print("1. Execution Times Chart...")
    plot_execution_times()

    print("2. Time vs Size Chart...")
    plot_time_vs_size()

    print("3. Success/Failure Analysis...")
    plot_success_failure()

    print("4. Success Rate Analysis...")
    plot_success_rate()

    print("5. Time vs Solved Status...")
    plot_time_vs_solved()

    print("All charts generated successfully!")


if __name__ == "__main__":
    main()
