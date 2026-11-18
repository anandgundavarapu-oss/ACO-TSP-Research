
import random
import math
import matplotlib.pyplot as plt

# ==================================================
# ANT COLONY OPTIMIZATION - TRAVELLING SALESMAN
# ==================================================

# 1. CITY LOCATIONS

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

number_of_cities = len(cities)


# 2. ACO PARAMETERS

number_of_ants = 20
number_of_iterations = 100

alpha = 1.0
beta = 2.0
evaporation = 0.5
Q = 100


# 3. CALCULATE DISTANCES

distance = [
    [0.0 for _ in range(number_of_cities)]
    for _ in range(number_of_cities)
]

for i in range(number_of_cities):
    for j in range(number_of_cities):

        x1, y1 = cities[i]
        x2, y2 = cities[j]

        distance[i][j] = math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )


# 4. INITIALIZE PHEROMONES

pheromone = [
    [1.0 for _ in range(number_of_cities)]
    for _ in range(number_of_cities)
]

best_route = None
best_distance = float("inf")

best_distance_history = []


# 5. SELECT THE NEXT CITY

def select_next_city(current_city, unvisited):

    probabilities = []

    for city in unvisited:

        pheromone_value = pheromone[current_city][city]

        heuristic_value = 1 / distance[current_city][city]

        probability = (
            (pheromone_value ** alpha) *
            (heuristic_value ** beta)
        )

        probabilities.append(probability)

    total = sum(probabilities)

    probabilities = [
        p / total for p in probabilities
    ]

    selected_city = random.choices(
        unvisited,
        weights=probabilities,
        k=1
    )[0]

    return selected_city


# 6. MAIN ACO ALGORITHM

for iteration in range(number_of_iterations):

    all_routes = []

    # Each ant constructs a route

    for ant in range(number_of_ants):

        start_city = random.randint(
            0, number_of_cities - 1
        )

        route = [start_city]

        unvisited = [
            city for city in range(number_of_cities)
            if city != start_city
        ]

        # Ant visits all remaining cities

        while unvisited:

            current_city = route[-1]

            next_city = select_next_city(
                current_city,
                unvisited
            )

            route.append(next_city)

            unvisited.remove(next_city)

        # Return to starting city

        route.append(start_city)

        # Calculate total route distance

        total_distance = 0

        for i in range(len(route) - 1):

            total_distance += distance[
                route[i]
            ][
                route[i + 1]
            ]

        all_routes.append((route, total_distance))

        # Update best route

        if total_distance < best_distance:

            best_distance = total_distance

            best_route = route.copy()


    # 7. PHEROMONE EVAPORATION

    for i in range(number_of_cities):

        for j in range(number_of_cities):

            pheromone[i][j] *= (1 - evaporation)


    # 8. PHEROMONE DEPOSITION

    for route, route_distance in all_routes:

        deposit = Q / route_distance

        for i in range(len(route) - 1):

            city_a = route[i]
            city_b = route[i + 1]

            pheromone[city_a][city_b] += deposit

            pheromone[city_b][city_a] += deposit


    # Record best distance in each iteration

    best_distance_history.append(best_distance)


    # Display progress

    if (iteration + 1) % 10 == 0:

        print(
            f"Iteration {iteration + 1}: "
            f"Best Distance = {best_distance:.2f}"
        )


# 9. DISPLAY FINAL RESULTS

print("\n========== ACO RESULTS ==========")

print("\nBest Route:")

print(" -> ".join(map(str, best_route)))

print(f"\nShortest Distance: {best_distance:.2f}")

print("\nCity Coordinates:")

for city in best_route:

    print(f"City {city}: {cities[city]}")

print("\n=================================")


# 10. GRAPH 1 - BEST ROUTE

x = [cities[city][0] for city in best_route]

y = [cities[city][1] for city in best_route]

plt.figure(figsize=(9, 6))

plt.plot(
    x,
    y,
    marker="o",
    linewidth=2,
    markersize=8
)

for city in range(number_of_cities):

    plt.annotate(
        f"City {city}",
        cities[city],
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.title("Ant Colony Optimization - Best Route")

plt.xlabel("X Coordinate")

plt.ylabel("Y Coordinate")

plt.grid(True)

plt.tight_layout()


# 11. GRAPH 2 - CONVERGENCE GRAPH

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, number_of_iterations + 1),
    best_distance_history,
    linewidth=2
)

plt.title("ACO - Convergence Graph")

plt.xlabel("Number of Iterations")

plt.ylabel("Best Route Distance")

plt.grid(True)

plt.tight_layout()


# 12. SHOW BOTH GRAPHS

plt.show()
