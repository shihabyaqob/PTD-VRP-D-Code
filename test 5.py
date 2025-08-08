import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB


# Constants
AREA_SIZE = 10.0  # km
N_DELIVERY_LOCATIONS = 50
DEPOT = (0, 0)
DRONE_ENDURANCE = 20  # minutes
N_DRONES = 10
TRUCK_SPEED = 60.0  # km/h
DRONE_SPEED = 40.0  # km/h
SERVICE_TIME = 5.0  # in minutes
np.random.seed()  # Set a specific seed for reproducibility

def generate_random_locations(n_locations, depot, range_km=AREA_SIZE):
    locations = np.random.uniform(-range_km, range_km, size=(n_locations, 2))
    return np.vstack([depot, locations])

def is_within_endurance(cluster_center, locations, drone_speed, service_time, endurance):
    distances = np.linalg.norm(locations - cluster_center, axis=1)
    round_trip_times = 2 * (distances / drone_speed) + (service_time / 60)  # Convert service time to hours
    return np.all(round_trip_times <= (endurance / 60))  # Convert endurance to hours

def validate_cluster_capacity(labels, max_capacity):
    unique, counts = np.unique(labels, return_counts=True)
    return np.all(counts <= max_capacity)

def simulate_disruptions(route, disruption_probability, disruption_type):
    adjusted_route = route.copy()
    disruption_indices = []
    
    for i in range(1, len(route) - 1):
        if np.random.rand() < disruption_probability:
            disruption_indices.append(i)
            if disruption_type == 'road_closure':
                # Skip the disrupted point
                adjusted_route[i] = adjusted_route[i-1]
    
    return adjusted_route, disruption_indices

def calculate_adjusted_delivery_time(route, locations, truck_speed, service_time, disruption_probability, disruption_type):
    adjusted_route, disruption_indices = simulate_disruptions(route, disruption_probability, disruption_type)
    adjusted_time, truck_time, service_time = calculate_tsp_delivery_time(adjusted_route, locations, truck_speed, service_time)
    return adjusted_time, truck_time, service_time, disruption_indices

def evaluate_route_robustness(locations, route, truck_speed, service_time, num_simulations, disruption_probability, disruption_type):
    adjusted_times = []
    disruption_counts = np.zeros(len(route))
    
    for _ in range(num_simulations):
        adjusted_time, _, _, disruption_indices = calculate_adjusted_delivery_time(route, locations, truck_speed, service_time, disruption_probability, disruption_type)
        adjusted_times.append(adjusted_time)
        for idx in disruption_indices:
            disruption_counts[idx] += 1
    
    mean_adjusted_time = np.mean(adjusted_times)
    std_adjusted_time = np.std(adjusted_times)
    
    return mean_adjusted_time, std_adjusted_time, disruption_counts


def cluster_locations(locations, max_drones_per_truck=N_DRONES, max_clusters=N_DELIVERY_LOCATIONS):
    n_clusters = 1
    while n_clusters <= max_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(locations)
        if all(is_within_endurance(center, locations[kmeans.labels_ == i], DRONE_SPEED, SERVICE_TIME, DRONE_ENDURANCE) for i, center in enumerate(kmeans.cluster_centers_)):
            print("Number of Clusters: ", n_clusters)
            print("Number of Delivery Locations in each Cluster: ", np.bincount(kmeans.labels_))
            return kmeans.labels_, kmeans.cluster_centers_
        n_clusters += 1
    raise ValueError("Failed to find a valid clustering configuration.")

def create_data_model(locations, metric):
    data = {}
    data['distance_matrix'] = cdist(locations, locations, metric=metric).tolist()
    data['num_vehicles'] = 1
    data['depot'] = 0 # The depot is the first location in the locations list
    return data

def solve_tsp_or_tools(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Scale the distances by 1000 and convert to integers
        return int(data['distance_matrix'][from_node][to_node] * 1000)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])  # Ensure it returns to the depot
        return route
    else:
        return None


def solve_tsp_with_gurobi(data):
    distance_matrix = data['distance_matrix']
    n = len(distance_matrix)  # Number of locations

    # Define your Gurobi license options
    options = {
        "WLSACCESSID": "95c3994f-513d-4b8c-a42c-0194dad8ec5c",  # Replace with your actual WLSACCESSID
        "WLSSECRET": "3dc0e444-1f74-49f3-96cc-e9abaabb8ce4",    # Replace with your actual WLSSECRET
        "LICENSEID": 2551395,                                    # Replace withzzz your actual LICENSEID
    }


    # Create a Gurobi environment with the specified options
    with gp.Env(params=options) as env, gp.Model(env=env) as model:
        # Set Gurobi parameters for optimization
        model.setParam('Threads', 16)         # Use 4 threads for parallel processing
        model.setParam('MIPGap', 0.01)       # Set the MIP gap for optimization
        model.setParam('LogToConsole', 0)  # Suppress Gurobi log output

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
        model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n)), name="outflow")
        model.addConstrs((gp.quicksum(x[j, i] for j in range(n) if j != i) == 1 for i in range(n)), name="inflow")
        u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
        model.addConstrs((u[i] >= 1 for i in range(1, n)))
        model.addConstrs((u[j] - u[i] + (n - 1) * x[i, j] <= n - 2 for i in range(1, n) for j in range(1, n) if i != j))
        model.optimize()

        if model.status == GRB.OPTIMAL:
            route = [0]  # Start at depot
            current_location = 0
            while len(route) < n:
                for j in range(n):
                    if x[current_location, j].X > 0.5:  # If the route goes from current_location to j
                        route.append(j)
                        current_location = j
                        break
            route.append(0)  # Return to depot
            return route
        else:
            return None


def calculate_tsp_delivery_time(route, locations, truck_speed, service_time):
    truck_distances = cdist(locations[route[:-1]], locations[route[1:]], metric='cityblock').diagonal()
    truck_time = np.sum(truck_distances) / truck_speed  # this is in hours
    #print ("Sum of Truck Distances: ", np.sum(truck_distances))
    #print ("Truck Time: ", truck_time)
    service_time_hours = service_time / 60  # converting minutes to hours
    total_time = truck_time + service_time_hours * (len(route) - 1)  # service at every stop except depot return

    return total_time, truck_time, service_time_hours * (len(route) - 1)

def calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, max_drones_per_cluster=N_DRONES):
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    
    tsp_route = solve_tsp_with_gurobi(create_data_model(cluster_centers_with_depot, metric='cityblock'))
    if tsp_route is None:
        return None, None

    truck_distances = cdist(cluster_centers_with_depot[tsp_route[:-1]], cluster_centers_with_depot[tsp_route[1:]], metric='cityblock').diagonal()
    truck_time = np.sum(truck_distances) / truck_speed  # in hours

    total_drone_time = 0
    total_utilization = 0
    cluster_count = 0  # To count the number of clusters processed

    for center_idx in tsp_route[1:-1]:
        cluster_locs = locations[labels == (center_idx - 1)]
        if cluster_locs.size == 0:
            continue

        distances = np.linalg.norm(cluster_locs - cluster_centers_with_depot[center_idx], axis=1)  # distance to each delivery point
        delivery_times = distances / drone_speed  # time to each delivery point
        round_trip_times = 2 * delivery_times   # out and back for each delivery point

        drone_end_times = np.zeros(max_drones_per_cluster)  # end times for each drone's deliveries
        for trip_time in round_trip_times:
            earliest_drone = np.argmin(drone_end_times)
            drone_end_times[earliest_drone] += trip_time
                    
        # Print statements for debugging (commented)
        # print("drone_end_times: ", drone_end_times)
        
        max_end_time = np.max(drone_end_times)  # max time a drone is busy
        total_drone_time += max_end_time
        
        each_drone_utilization = drone_end_times / max_end_time
        # print("each_drone_utilization: ", each_drone_utilization)
        
        cluster_utilization = np.sum(each_drone_utilization) / max_drones_per_cluster
        # print("cluster_utilization: ", cluster_utilization)
        
        total_utilization += cluster_utilization
        cluster_count += 1  # Increment cluster count

    if cluster_count > 0:
        total_utilization /= cluster_count  # Average utilization over all clusters
    else:
        total_utilization = 0  # Handle case where no clusters were processed

    # Print statements for debugging (commented)
    # print("total_drone_time: ", total_drone_time)
    # print("total_utilization: ", total_utilization)

    total_service_time = (service_time / 60) * (len(cluster_centers) - 1)  # in hours
    
    total_time = truck_time + total_drone_time + total_service_time
    
    return total_time, truck_time, total_drone_time, total_service_time, total_utilization

def calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
    optimized_weights_reshaped = optimized_weights.reshape(-1, 2)
    optimized_centers = cluster_centers + optimized_weights_reshaped
    return calculate_clustering_delivery_time(optimized_centers, locations, labels, drone_speed, truck_speed, service_time)


def optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
    num_centers = len(cluster_centers)
    initial_weights = np.zeros((num_centers, 2))

    def penalty_function(weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
        penalty = 0
        new_centers = cluster_centers + weights.reshape(len(cluster_centers), 2)
        for idx, center in enumerate(new_centers):
            if not is_within_endurance(center, locations[labels == idx], drone_speed, service_time, DRONE_ENDURANCE):
                penalty += 1000
        return calculate_optimize_weights_delivery_time(weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time)[0] + penalty

    bounds = [(-2, 2) for _ in range(num_centers * 2)]
    flat_initial_weights = initial_weights.flatten()

    result = minimize(
        penalty_function,
        flat_initial_weights,
        args=(cluster_centers, locations, labels, drone_speed, truck_speed, service_time),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    if result.success:
        return result.x.reshape(num_centers, 2)
    else:
        print("Optimization failed:", result.message)
        return initial_weights

import pandas as pd

def save_data_to_csv(filename, **data):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def evaluate_performance(n_iterations, drone_speed=40.0, truck_speed=60.0, service_time=5.0, disruption_probability=0.1, disruption_type='road_closure', num_robustness_simulations=100):
    Total_delivery_times_tsp = []
    Total_delivery_times_cluster = []
    Total_delivery_times_optimize = []
    
    Total_travel_times_tsp = []
    Total_travel_times_cluster = []
    Total_travel_times_optimize = []

    Total_truck_times_tsp = []
    Total_truck_times_cluster = []
    Total_truck_times_optimize = []

    Total_drone_times_cluster = []
    Total_drone_times_optimize = []

    Total_service_times_tsp = []
    Total_service_times_cluster = []
    Total_service_times_optimize = []

    Total_number_of_clusters = []

    Total_drone_utilization_cluster = []
    Total_drone_utilization_optimize = []

    Total_robustness_tsp = []
    Total_robustness_cluster = []
    Total_robustness_optimize = []

    for _ in range(n_iterations):
        print("Iteration:", _)
        locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
        labels, cluster_centers = cluster_locations(locations)
        Total_number_of_clusters.append(len(np.unique(labels)))

        tsp_route = solve_tsp_with_gurobi(create_data_model(locations, metric='cityblock'))
        if tsp_route:
            tsp_time, tsp_truck_time, tsp_service_time = calculate_tsp_delivery_time(tsp_route, locations, truck_speed, service_time)
            Total_delivery_times_tsp.append(tsp_time)
            Total_travel_times_tsp.append(tsp_truck_time)
            Total_truck_times_tsp.append(tsp_truck_time)
            Total_service_times_tsp.append(tsp_service_time)

            # Evaluate robustness for TSP
            mean_adjusted_time_tsp, std_adjusted_time_tsp, _ = evaluate_route_robustness(locations, tsp_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_tsp.append((mean_adjusted_time_tsp, std_adjusted_time_tsp))

        cluster_time, cluster_truck_time, cluster_drone_time, cluster_service_time, drone_utilizations = calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if cluster_time:
            Total_delivery_times_cluster.append(cluster_time)
            Total_travel_times_cluster.append(cluster_truck_time + cluster_drone_time)
            Total_truck_times_cluster.append(cluster_truck_time)
            Total_drone_times_cluster.append(cluster_drone_time)
            Total_service_times_cluster.append(cluster_service_time)
            Total_drone_utilization_cluster.append(drone_utilizations)

            # Evaluate robustness for Clustering
            cluster_route = solve_tsp_with_gurobi(create_data_model(np.vstack([DEPOT, cluster_centers]), metric='cityblock'))
            mean_adjusted_time_cluster, std_adjusted_time_cluster, _ = evaluate_route_robustness(np.vstack([DEPOT, cluster_centers]), cluster_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_cluster.append((mean_adjusted_time_cluster, std_adjusted_time_cluster))

        optimized_weights = optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if optimized_weights is not None:
            optimize_time, optimize_truck_time, optimize_drone_time, optimize_service_time, drone_utilizations_opt = calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
            Total_delivery_times_optimize.append(optimize_time)
            Total_travel_times_optimize.append(optimize_truck_time + optimize_drone_time)
            Total_truck_times_optimize.append(optimize_truck_time)
            Total_drone_times_optimize.append(optimize_drone_time)
            Total_service_times_optimize.append(optimize_service_time)
            Total_drone_utilization_optimize.append(drone_utilizations_opt)

            # Evaluate robustness for Optimized
            optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
            optimize_route = solve_tsp_with_gurobi(create_data_model(np.vstack([DEPOT, optimized_centers]), metric='cityblock'))
            mean_adjusted_time_optimize, std_adjusted_time_optimize, _ = evaluate_route_robustness(np.vstack([DEPOT, optimized_centers]), optimize_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_optimize.append((mean_adjusted_time_optimize, std_adjusted_time_optimize))

    save_data_to_csv('Output.csv', Iteration=np.arange(1, n_iterations + 1),
                    Total_delivery_times_tsp=Total_delivery_times_tsp,
                    Total_delivery_times_cluster=Total_delivery_times_cluster,
                    Total_delivery_times_optimize=Total_delivery_times_optimize,
                    Total_travel_times_tsp=Total_travel_times_tsp,
                    Total_travel_times_cluster=Total_travel_times_cluster,
                    Total_travel_times_optimize=Total_travel_times_optimize,
                    Total_truck_times_tsp=Total_truck_times_tsp,
                    Total_truck_times_cluster=Total_truck_times_cluster,
                    Total_truck_times_optimize=Total_truck_times_optimize,
                    Total_drone_times_cluster=Total_drone_times_cluster,
                    Total_drone_times_optimize=Total_drone_times_optimize,
                    Total_service_times_tsp=Total_service_times_tsp,
                    Total_service_times_cluster=Total_service_times_cluster,
                    Total_service_times_optimize=Total_service_times_optimize,
                    Drone_utilization_cluster=Total_drone_utilization_cluster,
                    Drone_utilization_optimize=Total_drone_utilization_optimize,
                    Number_of_clusters=Total_number_of_clusters,
                    Robustness_tsp=Total_robustness_tsp,
                    Robustness_cluster=Total_robustness_cluster,
                    Robustness_optimize=Total_robustness_optimize)

    print_statistics(Total_delivery_times_tsp, Total_delivery_times_cluster, Total_delivery_times_optimize, Total_drone_times_cluster, Total_drone_times_optimize)
    print_robustness_statistics(Total_robustness_tsp, Total_robustness_cluster, Total_robustness_optimize)

def print_robustness_statistics(robustness_tsp, robustness_cluster, robustness_optimize):
    print("\nRobustness Statistics:")
    print("TSP - Mean Adjusted Time: {:.2f}, Std Dev: {:.2f}".format(
        np.mean([r[0] for r in robustness_tsp]),
        np.mean([r[1] for r in robustness_tsp])
    ))
    print("Clustering - Mean Adjusted Time: {:.2f}, Std Dev: {:.2f}".format(
        np.mean([r[0] for r in robustness_cluster]),
        np.mean([r[1] for r in robustness_cluster])
    ))
    print("Optimized - Mean Adjusted Time: {:.2f}, Std Dev: {:.2f}".format(
        np.mean([r[0] for r in robustness_optimize]),
        np.mean([r[1] for r in robustness_optimize])
    ))

def plot_robustness_comparison(robustness_tsp, robustness_cluster, robustness_optimize):
    plt.figure(figsize=(12, 6))
    
    tsp_means = [r[0] for r in robustness_tsp]
    cluster_means = [r[0] for r in robustness_cluster]
    optimize_means = [r[0] for r in robustness_optimize]
    
    plt.boxplot([tsp_means, cluster_means, optimize_means], labels=['TSP', 'Clustering', 'Optimized'])
    plt.title('Robustness Comparison')
    plt.ylabel('Mean Adjusted Delivery Time (hours)')
    plt.show()


def print_statistics(total_tsp, total_cluster, total_optimize, total_drone_times_cluster, total_drone_times_optimize):
    print("Average Total Delivery Time (TSP):", np.mean(total_tsp))
    print("Average Total Delivery Time (Clustering):", np.mean(total_cluster))
    print("Average Total Delivery Time (Optimize Weights):", np.mean(total_optimize))
    print("Average Total Drone Time (Clustering):", np.mean(total_drone_times_cluster))
    print("Average Total Drone Time (Optimize Weights):", np.mean(total_drone_times_optimize))
    improvement_cluster = ((np.mean(total_tsp) - np.mean(total_cluster)) / np.mean(total_tsp)) * 100
    improvement_optimize = ((np.mean(total_cluster) - np.mean(total_optimize)) / np.mean(total_cluster)) * 100
    print("Improvement from TSP to Clustering:", improvement_cluster, "%")
    print("Improvement from Clustering to Optimize Weights:", improvement_optimize, "%")


def plot_routes(locations, labels, cluster_centers, tsp_route, initial_tsp_route=None, optimized_centers=None):
    plt.figure(figsize=(10, 8), dpi=200)

    # Plot delivery locations
    scatter1 = plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')

    # Plot initial TSP route if provided
    if initial_tsp_route is not None:
        initial_route = np.array([locations[i] for i in initial_tsp_route])
        line1, = plt.plot(initial_route[:, 0], initial_route[:, 1], 'r-', label='Initial TSP Route')

    # Plot cluster centers
    scatter2 = plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='green', marker='x', label='Cluster Centers')

    # Plot TSP route through cluster centers if provided
    if tsp_route is not None:
        tsp_route_coords = np.array([cluster_centers[i] for i in tsp_route])
        line2, = plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'g--', label='Cluster Centers TSP Route')

    # Plot optimized centers and route if provided
    if optimized_centers is not None:
        scatter3 = plt.scatter(optimized_centers[:, 0], optimized_centers[:, 1], c='purple', marker='s', label='Optimized Centers')
        if tsp_route is not None:
            optimized_route_coords = np.array([optimized_centers[i] for i in tsp_route])
            line3, = plt.plot(optimized_route_coords[:, 0], optimized_route_coords[:, 1], 'm--', label='Optimized Centers TSP Route')

    # Plot depot
    scatter4 = plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')

    # Add labels and title
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.title('Routes and Locations')

    # Create separate legends
    if initial_tsp_route is not None and tsp_route is not None:
        line_legend = plt.legend(handles=[line1, line2, line3], loc='upper left', bbox_to_anchor=(1, 1))
    else:
        line_legend = plt.legend(handles=[line1, line2, line3], loc='upper left', bbox_to_anchor=(1, 1))

    scatter_legend = plt.legend(handles=[scatter1, scatter2, scatter3, scatter4], loc='upper left', bbox_to_anchor=(1, 0.87))

    # Add the first legend back to the plot
    plt.gca().add_artist(line_legend)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_TSP_inital_route(locations, initial_tsp_route, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')
    initial_route_coords = np.array([locations[i] for i in initial_tsp_route])
    plt.plot(initial_route_coords[:, 0], initial_route_coords[:, 1], 'r-', label='Initial TSP Route')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.legend()
    plt.title('Initial TSP Route and Locations')
    plt.grid(True)
    plt.show()

def plot_cluster_route(locations, cluster_centers, tsp_route, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='green', marker='x', label='Cluster Centers')
    tsp_route_with_depot = [0] + [x + 1 for x in tsp_route[1:-1]] + [0]
    tsp_route_coords = np.array([locations[0]] + [cluster_centers[i - 1] for i in tsp_route_with_depot[1:-1]] + [locations[0]])
    if tsp_route is not None:
        plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'g--', label='Cluster Centers TSP Route')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.legend()
    plt.title('Cluster Routes and Locations')
    plt.grid(True)
    plt.show()

def plot_optimize_route(locations, optimized_centers, tsp_route, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')
    plt.scatter(optimized_centers[:, 0], optimized_centers[:, 1], c='purple', marker='s', label='Optimized Centers')
    tsp_route_with_depot = [0] + [x + 1 for x in tsp_route[1:-1]] + [0]
    tsp_route_coords = np.array([locations[0]] + [optimized_centers[i - 1] for i in tsp_route_with_depot[1:-1]] + [locations[0]])
    if tsp_route is not None:
        plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'm--', label='Optimized Centers TSP Route')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.legend()
    plt.title('Optimized Routes and Locations')
    plt.grid(True)
    plt.show()
# plot both cluster and optimized routes
def plot_cluster_optimize_route(locations, cluster_centers, optimized_centers, tsp_route, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='green', marker='x', label='Cluster Centers')
    plt.scatter(optimized_centers[:, 0], optimized_centers[:, 1], c='purple', marker='s', label='Optimized Centers')
    tsp_route_with_depot = [0] + [x + 1 for x in tsp_route[1:-1]] + [0]
    tsp_route_coords = np.array([locations[0]] + [cluster_centers[i - 1] for i in tsp_route_with_depot[1:-1]] + [locations[0]])
    if tsp_route is not None:
        plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'g--', label='Cluster Centers TSP Route')
        tsp_route_coords = np.array([locations[0]] + [optimized_centers[i - 1] for i in tsp_route_with_depot[1:-1]] + [locations[0]])
        plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'm--', label='Optimized Centers TSP Route')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.legend()
    plt.title('Cluster and Optimized Routes and Locations')
    plt.grid(True)
    plt.show()

def plot_delivery_times(n_iterations, M_TSP, M_k, M_opt):
    n_iterations = np.arange(1, len(M_TSP) + 1)
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.plot(n_iterations, M_TSP, 'r-', label='TSP', marker='o')
    plt.plot(n_iterations, M_k, 'g--', label='Clustering', marker='x')
    plt.plot(n_iterations, M_opt, 'b-.', label='Optimize Weights', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Total Delivery Time (hours)')
    plt.title('Total Delivery Time vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_travel_times(n_iterations, M_TSP, M_k, M_opt):
    n_iterations = np.arange(1, len(M_TSP) + 1)
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.plot(n_iterations, M_TSP, 'r-', label='TSP', marker='o')
    plt.plot(n_iterations, M_k, 'g--', label='Clustering', marker='x')
    plt.plot(n_iterations, M_opt, 'b-.', label='Optimize Weights', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Total Travel Time (hours)')
    plt.title('Total Travel Time vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_truck_times(n_iterations, M_TSP, M_k, M_opt):
    n_iterations = np.arange(1, len(M_TSP) + 1)
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.plot(n_iterations, M_TSP, 'r-', label='TSP', marker='o')
    plt.plot(n_iterations, M_k, 'g--', label='Clustering', marker='x')
    plt.plot(n_iterations, M_opt, 'b-.', label='Optimize Weights', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Total Truck Time (hours)')
    plt.title('Total Truck Time vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drone_times(n_iterations, M_k, M_opt):
    n_iterations = np.arange(1, len(M_k) + 1)
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.plot(n_iterations, M_k, 'g--', label='Clustering', marker='x')
    plt.plot(n_iterations, M_opt, 'b-.', label='Optimize Weights', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Total Drone Time (hours)')
    plt.title('Total Drone Time vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_travel_times_truck_drone(n_iterations, M_TSP, M_k, M_opt, M_k_drone, M_opt_drone):
    width = 0.25
    n_iterations = np.arange(1, len(M_TSP) + 1)
    plt.figure(figsize=(12, 6))
    
    # Truck times as bars
    bar1 = plt.bar(n_iterations - width, M_TSP, width, color='black', label='Truck TSP')
    bar2 = plt.bar(n_iterations, M_k, width, color='lightblue', label='Clustering Truck')
    bar3 = plt.bar(n_iterations + width, M_opt, width, color='red', label='Optimize Truck')

    # Drone times as lines
    line1, = plt.plot(n_iterations, M_k_drone, 'g--', label='Clustering Drones ')
    line2, = plt.plot(n_iterations, M_opt_drone, 'b-.', label='Optimize Drones ')

    plt.xlabel('Iteration')
    plt.ylabel('Travel Time (hours)')
    plt.title('Travel Time of Truck and Drones')
    
    # Create first legend for the truck
    truck_legend = plt.legend(handles=[bar1, bar2, bar3], loc='upper left', bbox_to_anchor=(1, 1))
    # Create second legend for the drones and add it to the plot
    plt.legend(handles=[line1, line2], loc='upper left', bbox_to_anchor=(1, 0.87))
    # Add the first legend back to the plot
    plt.gca().add_artist(truck_legend)

    plt.tight_layout()
    plt.show()


def plot_service_times(n_iterations, M_TSP, M_k, M_opt):
    n_iterations = np.arange(1, len(M_TSP) + 1)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(n_iterations, M_TSP, 'r-', label='TSP', marker='o')
    plt.plot(n_iterations, M_k, 'g--', label='Clustering', marker='x')
    plt.plot(n_iterations, M_opt, 'b-.', label='Optimize Weights', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Total Service Time (hours)')
    plt.title('Total Service Time vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_route_to_csv(locations, labels, cluster_centers, tsp_route, initial_tsp_route, optimized_centers):
    # Define the folder name
    folder_name = 'Route'
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save locations and labels
    locations_df = pd.DataFrame(locations, columns=['x', 'y'])
    locations_df['label'] = labels
    locations_df.to_csv(os.path.join(folder_name, 'locations.csv'), index=False)

    # Save cluster centers
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=['x', 'y'])
    cluster_centers_df.to_csv(os.path.join(folder_name, 'cluster_centers.csv'), index=False)

    # Save TSP routes
    tsp_route_df = pd.DataFrame({'tsp_route': tsp_route})
    tsp_route_df.to_csv(os.path.join(folder_name, 'tsp_route.csv'), index=False)

    initial_tsp_route_df = pd.DataFrame({'initial_tsp_route': initial_tsp_route})
    initial_tsp_route_df.to_csv(os.path.join(folder_name, 'initial_tsp_route.csv'), index=False)

    # Save optimized centers
    optimized_centers_df = pd.DataFrame(optimized_centers, columns=['x', 'y'])
    optimized_centers_df.to_csv(os.path.join(folder_name, 'optimized_centers.csv'), index=False)

def main():
    locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
    labels, cluster_centers = cluster_locations(locations)
    initial_data = create_data_model(locations, metric='cityblock')
    initial_tsp_route = solve_tsp_with_gurobi(initial_data)
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    data = create_data_model(cluster_centers_with_depot, metric='cityblock')
    tsp_route = solve_tsp_with_gurobi(data)

    if tsp_route and initial_tsp_route:
        optimized_weights = optimize_weights(cluster_centers, locations, labels, DRONE_SPEED, TRUCK_SPEED, SERVICE_TIME)
        if optimized_weights is not None:
            optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
            optimized_centers_with_depot = np.vstack([DEPOT, optimized_centers])
            
            save_route_to_csv(locations, labels, cluster_centers_with_depot, tsp_route, initial_tsp_route, optimized_centers_with_depot)

            # Uncomment to plot routes
            # plot_routes(locations, labels, cluster_centers_with_depot, tsp_route, initial_tsp_route=initial_tsp_route, optimized_centers=optimized_centers_with_depot)
            # plot_TSP_inital_route(locations, initial_tsp_route, labels)
            # plot_cluster_route(locations, cluster_centers_with_depot, tsp_route, labels)
            # plot_optimize_route(locations, optimized_centers_with_depot, tsp_route, labels)
            # plot_cluster_optimize_route(locations, cluster_centers_with_depot, optimized_centers_with_depot, tsp_route, labels)
            
            evaluate_performance(n_iterations=30, disruption_probability=0.1, disruption_type='road_closure', num_robustness_simulations=100)
    
        else:
            print("Optimization failed.")
    else:
        print("TSP solution not found.")

if __name__ == "__main__":
    main()


