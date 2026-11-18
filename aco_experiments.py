
import random
import math
import time
import csv
import itertools
import statistics
import matplotlib.pyplot as plt

# =====================================================
# ACO RESEARCH EXPERIMENTS
# Travelling Salesman Problem
# SRM Institute of Science and Technology
# =====================================================

# 1. CITY DATA

cities = {
    0: (0, 0),
    1: (2, 6),
    2: (5, 3),
    3: (6, 7),
    4: (8, 2),
    5: (1, 9),
    6: (7, 10),
    7: (10, 6)
}

N = len(cities)

distance = [
    [
        math.dist(cities[i], cities[j])
        for j in range(N)
    ]
    for i in range(N)
]

# 2. BASE PARAMETERS

BASE_ANTS = 20
BASE_ITERATIONS = 100
BASE_ALPHA = 1.0
BASE_BETA = 2.0
BASE_RHO = 0.5
Q = 100

REPEATS = 10


# 3. EXACT OPTIMAL SOLUTION
# Used as a reference for the 8-city problem.

def calculate_exact_solution():

    best_distance = float("inf")
    best_route = None

    for permutation in itertools.permutations(range(1, N)):

        route = (0,) + permutation + (0,)

        total = sum(
            distance[route[i]][route[i + 1]]
            for i in range(N)
        )

        if total < best_distance:
            best_distance = total
            best_route = route

    return best_route, best_distance


# 4. ANT COLONY OPTIMIZATION

def run_aco(
    ants,
    iterations,
    rho,
    seed,
    alpha=BASE_ALPHA,
    beta=BASE_BETA
):

    rng = random.Random(seed)

    pheromone = [
        [1.0 for _ in range(N)]
        for _ in range(N)
    ]

    best_route = None
    best_distance = float("inf")
    history = []

    start_time = time.perf_counter()

    for iteration in range(iterations):

        all_routes = []

        for ant in range(ants):

            start = rng.randrange(N)

            route = [start]

            unvisited = set(range(N))
            unvisited.remove(start)

            while unvisited:

                current = route[-1]

                choices = sorted(unvisited)

                weights = []

                for city in choices:

                    tau = pheromone[current][city]
                    eta = 1.0 / distance[current][city]

                    weight = (
                        (tau ** alpha) *
                        (eta ** beta)
                    )

                    weights.append(weight)

                next_city = rng.choices(
                    choices,
                    weights=weights,
                    k=1
                )[0]

                route.append(next_city)
                unvisited.remove(next_city)

            route.append(start)

            route_distance = sum(
                distance[route[i]][route[i + 1]]
                for i in range(N)
            )

            all_routes.append((route, route_distance))

            if route_distance < best_distance:

                best_distance = route_distance
                best_route = route.copy()

        # PHEROMONE EVAPORATION

        for i in range(N):
            for j in range(N):

                pheromone[i][j] *= (1 - rho)

        # PHEROMONE DEPOSITION

        for route, route_distance in all_routes:

            deposit = Q / route_distance

            for i in range(N):

                a = route[i]
                b = route[i + 1]

                pheromone[a][b] += deposit
                pheromone[b][a] += deposit

        history.append(best_distance)

    execution_time = time.perf_counter() - start_time

    return (
        best_route,
        best_distance,
        history,
        execution_time
    )


# 5. NEAREST-NEIGHBOUR BASELINE

def nearest_neighbour():

    best_route = None
    best_distance = float("inf")

    for start in range(N):

        route = [start]

        unvisited = set(range(N))
        unvisited.remove(start)

        while unvisited:

            current = route[-1]

            next_city = min(
                unvisited,
                key=lambda city: distance[current][city]
            )

            route.append(next_city)
            unvisited.remove(next_city)

        route.append(start)

        total = sum(
            distance[route[i]][route[i + 1]]
            for i in range(N)
        )

        if total < best_distance:

            best_distance = total
            best_route = route.copy()

    return best_route, best_distance


# 6. EXPERIMENT CONFIGURATIONS

experiments = {

    "Ant Population": [
        {"ants": 5, "iterations": 100, "rho": 0.5},
        {"ants": 10, "iterations": 100, "rho": 0.5},
        {"ants": 20, "iterations": 100, "rho": 0.5},
        {"ants": 40, "iterations": 100, "rho": 0.5}
    ],

    "Iteration Count": [
        {"ants": 20, "iterations": 50, "rho": 0.5},
        {"ants": 20, "iterations": 100, "rho": 0.5},
        {"ants": 20, "iterations": 200, "rho": 0.5},
        {"ants": 20, "iterations": 500, "rho": 0.5}
    ],

    "Evaporation Rate": [
        {"ants": 20, "iterations": 100, "rho": 0.1},
        {"ants": 20, "iterations": 100, "rho": 0.3},
        {"ants": 20, "iterations": 100, "rho": 0.5},
        {"ants": 20, "iterations": 100, "rho": 0.7}
    ]
}


# 7. RUN EXPERIMENTS

print("\n====================================")
print(" ANT COLONY OPTIMIZATION RESEARCH")
print("====================================")

exact_route, exact_distance = calculate_exact_solution()

nn_route, nn_distance = nearest_neighbour()

print(f"\nExact optimal distance: {exact_distance:.4f}")
print(f"Nearest-neighbour distance: {nn_distance:.4f}")

results = []
baseline_histories = []

experiment_number = 0

for experiment_name, configurations in experiments.items():

    print(f"\n--- {experiment_name} ---")

    for config in configurations:

        experiment_number += 1

        scores = []
        times = []
        best_run_route = None
        best_run_history = None

        for repeat in range(REPEATS):

            route, score, history, elapsed = run_aco(
                ants=config["ants"],
                iterations=config["iterations"],
                rho=config["rho"],
                seed=1000 + experiment_number * 100 + repeat
            )

            scores.append(score)
            times.append(elapsed)

            if best_run_route is None or score < min(scores[:-1], default=float("inf")):

                best_run_route = route
                best_run_history = history

        avg_score = statistics.mean(scores)
        best_score = min(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        avg_time = statistics.mean(times)

        gap = (
            (avg_score - exact_distance) /
            exact_distance
        ) * 100

        row = {
            "Experiment": experiment_name,
            "Ants": config["ants"],
            "Iterations": config["iterations"],
            "Evaporation": config["rho"],
            "Best Distance": best_score,
            "Average Distance": avg_score,
            "Standard Deviation": std_score,
            "Average Time (seconds)": avg_time,
            "Average Optimality Gap (%)": gap
        }

        results.append(row)

        print(
            f"Ants={config['ants']}, "
            f"Iterations={config['iterations']}, "
            f"Rho={config['rho']}: "
            f"Best={best_score:.4f}, "
            f"Average={avg_score:.4f}"
        )

        if (
            config["ants"] == BASE_ANTS
            and config["iterations"] == BASE_ITERATIONS
            and config["rho"] == BASE_RHO
            and experiment_name == "Ant Population"
        ):

            baseline_histories.append(best_run_history)


# 8. SAVE RESULTS TO CSV

with open(
    "aco_experiment_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# 9. SAVE EXACT AND BASELINE COMPARISON

print("\n========== BASELINE COMPARISON ==========")

print(f"Exact optimal distance: {exact_distance:.4f}")

print(f"Nearest-neighbour distance: {nn_distance:.4f}")


# 10. CONVERGENCE GRAPH

if baseline_histories:

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, BASE_ITERATIONS + 1),
        baseline_histories[0],
        label="ACO Best Distance"
    )

    plt.axhline(
        exact_distance,
        linestyle="--",
        label="Exact Optimal Distance"
    )

    plt.title("ACO Convergence Behaviour")

    plt.xlabel("Iterations")
    plt.ylabel("Route Distance")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("aco_convergence.png", dpi=300)


# 11. EXPERIMENT COMPARISON GRAPHS

for experiment_name in experiments:

    subset = [
        row for row in results
        if row["Experiment"] == experiment_name
    ]

    labels = []

    if experiment_name == "Ant Population":
        labels = [str(row["Ants"]) for row in subset]
        xlabel = "Number of Ants"

    elif experiment_name == "Iteration Count":
        labels = [str(row["Iterations"]) for row in subset]
        xlabel = "Number of Iterations"

    else:
        labels = [str(row["Evaporation"]) for row in subset]
        xlabel = "Evaporation Rate"

    averages = [
        row["Average Distance"]
        for row in subset
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        labels,
        averages,
        marker="o"
    )

    plt.title(f"ACO - {experiment_name}")

    plt.xlabel(xlabel)
    plt.ylabel("Average Route Distance")

    plt.grid(True)
    plt.tight_layout()

    filename = experiment_name.lower().replace(" ", "_")

    plt.savefig(
        f"aco_{filename}.png",
        dpi=300
    )


# 12. BEST ROUTE GRAPH

best_overall = min(
    results,
    key=lambda row: row["Best Distance"]
)

best_route, best_score, _, _ = run_aco(
    ants=best_overall["Ants"],
    iterations=best_overall["Iterations"],
    rho=best_overall["Evaporation"],
    seed=9999
)

# This final independent run is for visualization.
# Its result need not equal the best recorded experimental run.

x = [cities[city][0] for city in best_route]
y = [cities[city][1] for city in best_route]

plt.figure(figsize=(8, 6))

plt.plot(
    x,
    y,
    marker="o",
    linewidth=2
)

for city in range(N):

    plt.annotate(
        f"City {city}",
        cities[city],
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("ACO - Example Best Route")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.grid(True)
plt.tight_layout()

plt.savefig("aco_best_route.png", dpi=300)


# 13. FINAL SUMMARY

print("\n====================================")
print(" EXPERIMENTS COMPLETED")
print("====================================")

print("Number of experimental configurations:", len(results))
print("Runs per configuration:", REPEATS)

print("\nFiles generated:")

print("1. aco_experiment_results.csv")
print("2. aco_convergence.png")
print("3. aco_ant_population.png")
print("4. aco_iteration_count.png")
print("5. aco_evaporation_rate.png")
print("6. aco_best_route.png")

print("\nAll experiments completed successfully!")
