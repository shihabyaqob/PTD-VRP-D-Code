# =========================
# Import Libraries
# =========================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# =========================
# Constants
# =========================
AREA_SIZE = 10.0  # km
N_DELIVERY_LOCATIONS = 50
DEPOT = (0, 0)
DRONE_ENDURANCE = 20  # minutes
N_DRONES = 10
TRUCK_SPEED = 60.0  # km/h
DRONE_SPEED = 40.0  # km/h
SERVICE_TIME = 5.0  # in minutes
np.random.seed()  # Set a specific seed for reproducibility

# Time-Dependent Speed Profiles (Example)
TIME_WINDOWS = [
    (0, 6, 70.0),   # 12 AM - 6 AM: 70 km/h O 
    (6, 9, 40.0),   # 6 AM - 9 AM: 40 km/h
    (9, 16, 60.0),  # 9 AM - 4 PM: 60 km/h
    (16, 19, 35.0), # 4 PM - 7 PM: 35 km/h
    (19, 24, 70.0)  # 7 PM - 12 AM: 70 km/h
]

START_TIME = 8.0  # Start time in hours (e.g., 8.0 for 8 AM)

# =========================
# Utility Functions
# =========================

def generate_random_locations(n_locations, depot, range_km=AREA_SIZE):
    """Generate random delivery locations within a specified range."""
    locations = np.random.uniform(-range_km, range_km, size=(n_locations, 2))
    return np.vstack([depot, locations])

def is_within_endurance(cluster_center, locations, drone_speed, service_time, endurance):
    """Check if all locations are within the drone's endurance limit."""
    distances = np.linalg.norm(locations - cluster_center, axis=1)
    round_trip_times = 2 * (distances / drone_speed) + (service_time / 60)  # Convert service time to hours
    return np.all(round_trip_times <= (endurance / 60))  # Convert endurance to hours

def validate_cluster_capacity(labels, max_capacity):
    """Validate that no cluster exceeds its capacity."""
    unique, counts = np.unique(labels, return_counts=True)
    return np.all(counts <= max_capacity)

def get_truck_speed_at_time(current_time):
    """Get the truck's speed based on the current time of day."""
    # current_time is in hours (e.g., 8.5 for 8:30 AM)
    for start, end, speed in TIME_WINDOWS:
        if start <= current_time % 24 < end:
            return speed
    return TRUCK_SPEED  # Default speed if no time window matches

def calculate_time_dependent_travel_time(distance, departure_time):
    """Calculate travel time considering time-dependent speeds."""
    time = 0
    remaining_distance = distance
    current_time = departure_time

    while remaining_distance > 0:
        speed = get_truck_speed_at_time(current_time)
        time_interval = 0.1  # Small time increment in hours (e.g., 6 minutes)
        distance_covered = speed * time_interval
        if distance_covered > remaining_distance:
            time += remaining_distance / speed
            remaining_distance = 0
        else:
            time += time_interval
            remaining_distance -= distance_covered
            current_time += time_interval

    return time

# =========================
# Clustering and TSP Functions
# =========================

def cluster_locations(locations, max_drones_per_truck=N_DRONES, max_clusters=N_DELIVERY_LOCATIONS):
    """Cluster locations using KMeans and validate with drone endurance constraints."""
    n_clusters = 1
    while n_clusters <= max_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(locations)
        if all(is_within_endurance(center, locations[kmeans.labels_ == i], DRONE_SPEED, SERVICE_TIME, DRONE_ENDURANCE) for i, center in enumerate(kmeans.cluster_centers_)):
            print("Number of Clusters:", n_clusters)
            print("Number of Delivery Locations in each Cluster:", np.bincount(kmeans.labels_))
            return kmeans.labels_, kmeans.cluster_centers_
        n_clusters += 1
    raise ValueError("Failed to find a valid clustering configuration.")

# =========================
# TSP Solvers
# =========================

def create_time_dependent_data_model(locations, departure_times):
    """Create a data model for the TSP solver with time-dependent travel times."""
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    travel_time_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                travel_time_matrix[i][j] = 0
            else:
                distance = np.linalg.norm(locations[i] - locations[j], ord=1)  # Manhattan distance
                departure_time = departure_times[i]
                travel_time = calculate_time_dependent_travel_time(distance, departure_time)
                travel_time_matrix[i][j] = travel_time

    data = {}
    data['time_matrix'] = travel_time_matrix.tolist()
    data['num_vehicles'] = 1
    data['depot'] = 0  # The depot is the first location in the locations list
    return data

def solve_tsp_with_gurobi_time_windows(data, service_times):
    """Solve the TSP using Gurobi with time windows."""
    time_matrix = data['time_matrix']
    n = len(time_matrix)  # Number of locations

    # Define Gurobi model
    model = gp.Model()

    # Suppress Gurobi output
    model.Params.LogToConsole = 0

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    arrival_times = model.addVars(n, vtype=GRB.CONTINUOUS, name="arrival_times")

    # Objective: Minimize total time (arrival time back at depot)
    model.setObjective(arrival_times[0], GRB.MINIMIZE)

    # Constraints
    # Each node must be entered and left exactly once
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n)), name="outflow")
    model.addConstrs((gp.quicksum(x[j, i] for j in range(n) if j != i) == 1 for i in range(n)), name="inflow")

    # Subtour elimination constraints
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n, name="u")
    model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1 for i in range(n) for j in range(n) if i != j and (i != 0 and j != 0)), name="subtour")

    # Time window constraints
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(arrival_times[j] >= (arrival_times[i] + service_times[i] + time_matrix[i][j] - (1 - x[i, j]) * 1e5), name=f"time_{i}_{j}")

    # Set starting time at depot
    model.addConstr(arrival_times[0] == START_TIME, name="start_time")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        route = [0]  # Start at depot
        current_location = 0
        while len(route) < n:
            for j in range(n):
                if x[current_location, j].X > 0.5:
                    route.append(j)
                    current_location = j
                    break
        # Retrieve arrival times
        arrival_times_list = [arrival_times[i].X for i in range(n)]
        return route, arrival_times_list
    else:
        return None, None

# =========================
# Delivery Time Calculations
# =========================

def calculate_clustering_delivery_time_time_dependent(cluster_centers, locations, labels, drone_speed, service_time, max_drones_per_cluster=N_DRONES):
    """Calculate the total delivery time based on clustering with time-dependent travel times."""
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    n_clusters = len(cluster_centers) + 1  # Including depot
    service_times = [0] + [service_time / 60] * (n_clusters - 1)  # Service time in hours

    # Initial departure times (starting times) for each location (initialize with START_TIME)
    departure_times = [START_TIME] * n_clusters

    data = create_time_dependent_data_model(cluster_centers_with_depot, departure_times)
    tsp_route, arrival_times = solve_tsp_with_gurobi_time_windows(data, service_times)

    if tsp_route is None:
        return None, None

    total_truck_time = arrival_times[0] - START_TIME  # Total time from start to return to depot
    total_service_time = sum(service_times[1:])  # Exclude depot service time

    total_drone_time = 0
    total_utilization = 0
    cluster_count = 0

    for idx, center_idx in enumerate(tsp_route[1:-1], start=1):
        cluster_arrival_time = arrival_times[center_idx]
        cluster_locs = locations[labels == (center_idx - 1)]
        if cluster_locs.size == 0:
            continue

        distances = np.linalg.norm(cluster_locs - cluster_centers_with_depot[center_idx], axis=1)
        delivery_times = distances / drone_speed
        round_trip_times = 2 * delivery_times + (service_time / 60)

        drone_end_times = np.zeros(max_drones_per_cluster)
        for trip_time in round_trip_times:
            earliest_drone = np.argmin(drone_end_times)
            drone_end_times[earliest_drone] += trip_time

        max_end_time = np.max(drone_end_times)
        total_drone_time += max_end_time

        each_drone_utilization = drone_end_times / max_end_time
        cluster_utilization = np.sum(each_drone_utilization) / max_drones_per_cluster

        total_utilization += cluster_utilization
        cluster_count += 1

    if cluster_count > 0:
        total_utilization /= cluster_count
    else:
        total_utilization = 0

    total_time = total_truck_time + total_drone_time

    return total_time, total_truck_time, total_drone_time, total_service_time, total_utilization, arrival_times

def calculate_optimize_weights_delivery_time_time_dependent(optimized_weights, cluster_centers, locations, labels, drone_speed, service_time):
    """Calculate the delivery time using optimized cluster centers with time-dependent travel times."""
    optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
    return calculate_clustering_delivery_time_time_dependent(optimized_centers, locations, labels, drone_speed, service_time)

# =========================
# Optimization Functions
# =========================

def optimize_weights_time_dependent(cluster_centers, locations, labels, drone_speed, service_time):
    """Optimize the cluster centers to minimize delivery time with time-dependent travel times."""
    num_centers = len(cluster_centers)
    initial_weights = np.zeros((num_centers, 2))

    def penalty_function(weights, cluster_centers, locations, labels, drone_speed, service_time):
        penalty = 0
        new_centers = cluster_centers + weights.reshape(len(cluster_centers), 2)
        for idx, center in enumerate(new_centers):
            if not is_within_endurance(center, locations[labels == idx], drone_speed, service_time, DRONE_ENDURANCE):
                penalty += 1000
        total_time = calculate_optimize_weights_delivery_time_time_dependent(weights, cluster_centers, locations, labels, drone_speed, service_time)[0]
        return total_time + penalty

    bounds = [(-2, 2) for _ in range(num_centers * 2)]
    flat_initial_weights = initial_weights.flatten()

    result = minimize(
        penalty_function,
        flat_initial_weights,
        args=(cluster_centers, locations, labels, drone_speed, service_time),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    if result.success:
        return result.x.reshape(num_centers, 2)
    else:
        print("Optimization failed:", result.message)
        return initial_weights

# =========================
# Performance Evaluation
# =========================

def evaluate_performance_time_dependent(n_iterations, drone_speed=40.0, service_time=5.0):
    """Evaluate the performance over multiple iterations with time-dependent travel times."""
    Total_delivery_times_cluster = []
    Total_delivery_times_optimize = []

    Total_truck_times_cluster = []
    Total_truck_times_optimize = []

    Total_drone_times_cluster = []
    Total_drone_times_optimize = []

    Total_service_times_cluster = []
    Total_service_times_optimize = []

    Total_drone_utilization_cluster = []
    Total_drone_utilization_optimize = []

    for i in range(n_iterations):
        print("Iteration:", i)
        locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
        labels, cluster_centers = cluster_locations(locations)

        # Clustering without optimization
        cluster_time, cluster_truck_time, cluster_drone_time, cluster_service_time, drone_utilizations, _ = calculate_clustering_delivery_time_time_dependent(cluster_centers, locations, labels, drone_speed, service_time)
        if cluster_time:
            Total_delivery_times_cluster.append(cluster_time)
            Total_truck_times_cluster.append(cluster_truck_time)
            Total_drone_times_cluster.append(cluster_drone_time)
            Total_service_times_cluster.append(cluster_service_time)
            Total_drone_utilization_cluster.append(drone_utilizations)

        # Clustering with optimized weights
        optimized_weights = optimize_weights_time_dependent(cluster_centers, locations, labels, drone_speed, service_time)
        if optimized_weights is not None:
            optimize_time, optimize_truck_time, optimize_drone_time, optimize_service_time, drone_utilizations_opt, _ = calculate_optimize_weights_delivery_time_time_dependent(optimized_weights, cluster_centers, locations, labels, drone_speed, service_time)
            Total_delivery_times_optimize.append(optimize_time)
            Total_truck_times_optimize.append(optimize_truck_time)
            Total_drone_times_optimize.append(optimize_drone_time)
            Total_service_times_optimize.append(optimize_service_time)
            Total_drone_utilization_optimize.append(drone_utilizations_opt)

    # Print statistics
    print("Average Total Delivery Time (Clustering):", np.mean(Total_delivery_times_cluster))
    print("Average Total Delivery Time (Optimized):", np.mean(Total_delivery_times_optimize))
    improvement = ((np.mean(Total_delivery_times_cluster) - np.mean(Total_delivery_times_optimize)) / np.mean(Total_delivery_times_cluster)) * 100
    print("Improvement from Clustering to Optimized:", improvement, "%")

# =========================
# Main Execution
# =========================

# Main function
def main():
    """Main function to run the simulation and evaluate performance with time-dependent travel times."""
    locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
    labels, cluster_centers = cluster_locations(locations)
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])

    # Clustering without optimization
    cluster_time, cluster_truck_time, cluster_drone_time, cluster_service_time, drone_utilizations, arrival_times = calculate_clustering_delivery_time_time_dependent(cluster_centers, locations, labels, DRONE_SPEED, SERVICE_TIME)
    print("Total Delivery Time (Clustering):", cluster_time)

    # Clustering with optimized weights
    optimized_weights = optimize_weights_time_dependent(cluster_centers, locations, labels, DRONE_SPEED, SERVICE_TIME)
    if optimized_weights is not None:
        optimize_time, optimize_truck_time, optimize_drone_time, optimize_service_time, drone_utilizations_opt, arrival_times_opt = calculate_optimize_weights_delivery_time_time_dependent(optimized_weights, cluster_centers, locations, labels, DRONE_SPEED, SERVICE_TIME)
        print("Total Delivery Time (Optimized):", optimize_time)
    else:
        print("Optimization failed.")

    # Evaluate performance over multiple iterations
    evaluate_performance_time_dependent(n_iterations=10)

if __name__ == "__main__":
    main()