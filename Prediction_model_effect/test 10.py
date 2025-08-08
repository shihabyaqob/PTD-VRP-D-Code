# =========================
# Import Libraries
# =========================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
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
DEFAULT_TRUCK_SPEED = 60.0  # km/h
DRONE_SPEED = 40.0  # km/h
SERVICE_TIME = 5.0  # in minutes
np.random.seed()  # Set a specific seed for reproducibility
  

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


def simulate_disruptions(route, disruption_probability, disruption_type):
    """Simulate disruptions on a given route."""
    adjusted_route = route.copy()
    disruption_indices = []
    
    for i in range(1, len(route) - 1):
        if np.random.rand() < disruption_probability:
            disruption_indices.append(i)
            if disruption_type == 'road_closure':
                adjusted_route[i] = adjusted_route[i - 1]
    
    return adjusted_route, disruption_indices


def calculate_adjusted_delivery_time(route, locations, truck_speed, service_time, disruption_probability, disruption_type):
    """Calculate adjusted delivery time after accounting for disruptions."""
    adjusted_route, disruption_indices = simulate_disruptions(route, disruption_probability, disruption_type)
    adjusted_time, truck_time, service_time = calculate_tsp_delivery_time(adjusted_route, locations, truck_speed, service_time)
    return adjusted_time, truck_time, service_time, disruption_indices


def evaluate_route_robustness(locations, route, truck_speed, service_time, num_simulations, disruption_probability, disruption_type):
    """Evaluate the robustness of a route under simulated disruptions."""
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


def create_data_model(locations, metric):
    """Create a data model for the TSP solver."""
    data = {}
    data['distance_matrix'] = cdist(locations, locations, metric=metric).tolist()
    data['num_vehicles'] = 1
    data['depot'] = 0  # The depot is the first location in the locations list
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
    """Solve the TSP using Gurobi."""
    distance_matrix = data['distance_matrix']
    n = len(distance_matrix)  # Number of locations

    # Define Gurobi license options
    options = {
        "WLSACCESSID": "95c3994f-513d-4b8c-a42c-0194dad8ec5c",  # Replace with your actual WLSACCESSID
        "WLSSECRET": "3dc0e444-1f74-49f3-96cc-e9abaabb8ce4",    # Replace with your actual WLSSECRET
        "LICENSEID": 2551395,  # Replace with your actual LICENSEID
    }

    # Create Gurobi environment with the specified options
    with gp.Env(empty=True) as env:
        env.setParam('WLSACCESSID', options['WLSACCESSID'])
        env.setParam('WLSSECRET', options['WLSSECRET'])
        env.setParam('LICENSEID', options['LICENSEID'])
        env.setParam('LogToConsole', 0)  # Suppress console output
        env.start()

        with gp.Model(env=env) as model:
            # Set Gurobi parameters for optimization
            model.setParam('Threads', 16)      # Use 16 threads for parallel processing
            model.setParam('MIPGap', 0.01)     # Set the MIP gap for optimization

            # Define decision variables
            x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
            model.setObjective(
                gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), 
                GRB.MINIMIZE
            )
            model.addConstrs(
                (gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n)), 
                name="outflow"
            )
            model.addConstrs(
                (gp.quicksum(x[j, i] for j in range(n) if j != i) == 1 for i in range(n)), 
                name="inflow"
            )

            # Subtour elimination constraints
            u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
            model.addConstrs((u[i] >= 1 for i in range(1, n)), name="u_lb")
            model.addConstrs(
                (u[j] - u[i] + (n - 1) * x[i, j] <= n - 2 
                 for i in range(1, n) for j in range(1, n) if i != j), 
                name="subtour_elimination"
            )

            # Optimize the model
            model.optimize()

            # Extract the optimal route if the solution is found
            if model.status == GRB.OPTIMAL:
                route = [0]  # Start at depot
                current_location = 0
                while len(route) < n:
                    for j in range(n):
                        if x[current_location, j].X > 0.5:  # If route goes from current_location to j
                            route.append(j)
                            current_location = j
                            break
                route.append(0)  # Return to depot
                return route
            else:
                return None


# =========================
# Delivery Time Calculations
# =========================

def calculate_tsp_delivery_time(route, locations, truck_speed, service_time):
    """Calculate the total delivery time for the TSP route."""
    truck_distances = cdist(locations[route[:-1]], locations[route[1:]], metric='cityblock').diagonal()
    truck_time = np.sum(truck_distances) / truck_speed  # This is in hours
    service_time_hours = service_time / 60  # Convert service time to hours
    total_time = truck_time + service_time_hours * (len(route) - 1)  # Service at every stop except depot return
    return total_time, truck_time, service_time_hours * (len(route) - 1)

def get_truck_speed_at_time(delivery_time):
    """
    Load the 'Predicted_Speeds.csv' file, find the closest 15-minute interval for the given delivery time,
    and return the corresponding truck speed.
    
    Parameters:
        delivery_time (float): The time of delivery in hours (e.g., 9.75 for 9:45 AM).
    
    Returns:
        float: The predicted truck speed at the given delivery time.
    """
    # Load the CSV file
    try:
        speed_df = pd.read_csv('Predicted Speed/Predicted_Speeds.csv')
    except FileNotFoundError:
        raise FileNotFoundError("The 'Predicted_Speeds.csv' file was not found. Please check the file path.")
    
    # Ensure that the columns are stripped of any leading/trailing whitespace
    speed_df.columns = speed_df.columns.str.strip()
    
    # Check if the necessary columns exist in the file
    if 'Time of Day' not in speed_df.columns or 'Predicted Speed' not in speed_df.columns:
        raise KeyError("The 'Predicted_Speeds.csv' file must contain 'Time of Day' and 'Predicted Speed' columns.")
    
    # Ensure the data is sorted by 'Time of Day'
    speed_df = speed_df.sort_values('Time of Day')
    
    # Extract the time and speed values
    time_of_day_values = speed_df['Time of Day'].values
    predicted_speed_values = speed_df['Predicted Speed'].values

    # Create an interpolation function to handle time intervals accurately
    speed_interpolation = interp1d(time_of_day_values, predicted_speed_values, kind='linear', fill_value="extrapolate")
    
    # Normalize the delivery time to a 24-hour format (e.g., 25.75 -> 1.75 for next day)
    delivery_time = delivery_time % 24
    
    # Find the speed corresponding to the delivery time using interpolation
    truck_speed = float(speed_interpolation(delivery_time))
    
    return truck_speed

def calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, current_time, max_drones_per_cluster=N_DRONES):
    """Calculate the total delivery time based on clustering, optimized for time-dependent speeds."""
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    tsp_route = solve_tsp_with_gurobi(create_data_model(cluster_centers_with_depot, metric='cityblock'))
    if tsp_route is None:
        return None, None

    truck_distances = cdist(cluster_centers_with_depot[tsp_route[:-1]], cluster_centers_with_depot[tsp_route[1:]], metric='cityblock').diagonal()
    
    total_truck_time = 0
    total_drone_time = 0
    total_utilization = 0
    cluster_count = 0  # To count the number of clusters processed
    
    arrival_times = []  # List to store the arrival times at each cluster center

    for i, center_idx in enumerate(tsp_route[1:-1]):
        travel_distance = truck_distances[i]
        remaining_distance = travel_distance
        segment_travel_time = 0

        # Continue until the truck reaches the destination, crossing time windows if necessary
        while remaining_distance > 0:
            truck_speed = get_truck_speed_at_time(current_time)  # Get the speed for the current time
            ##################################################

            # Extract hours and minutes without changing current_time
            hours = int(current_time)  # Extract the hour part
            minutes = int((current_time - hours) * 60)  # Extract the minute part

            # Display the original current_time in HH:MM format
            # print("Truck Speed at Time {:02d}:{:02d}: {:.2f} km/h".format(hours, minutes, truck_speed))
            ##################################################

            travel_time = remaining_distance / truck_speed  # Time needed to cover the remaining distance
            current_time += travel_time
            segment_travel_time += travel_time
            remaining_distance = 0  # Since we calculate full travel in one go, set remaining distance to 0

        total_truck_time += segment_travel_time
        arrival_times.append(round(current_time, 4))  # Store arrival times after each segment

        # Calculate drone deliveries for this cluster
        cluster_locs = locations[labels == (center_idx - 1)]
        if cluster_locs.size == 0:
            continue

        distances = np.linalg.norm(cluster_locs - cluster_centers_with_depot[center_idx], axis=1)  # Distance to each delivery point
        delivery_times = distances / drone_speed  # Time to each delivery point
        drone_handling_time = 0.001 / 60 # Time to handle each delivery point (in hours)
        round_trip_times = 2 * delivery_times + drone_handling_time  # Out and back for each delivery point

        drone_end_times = np.zeros(max_drones_per_cluster)  # End times for each drone's deliveries
        for trip_time in round_trip_times:
            earliest_drone = np.argmin(drone_end_times)
            drone_end_times[earliest_drone] += trip_time
                    
        max_end_time = np.max(drone_end_times)  # Max time a drone is busy
        total_drone_time += max_end_time
        
        each_drone_utilization = drone_end_times / max_end_time  # Drone utilization
        cluster_utilization = np.sum(each_drone_utilization) / max_drones_per_cluster
        
        total_utilization += cluster_utilization
        cluster_count += 1  # Increment cluster count

    if cluster_count > 0:
        total_utilization /= cluster_count  # Average utilization over all clusters
    else:
        total_utilization = 0  # Handle case where no clusters were processed

    total_service_time = (service_time / 60) * (len(cluster_centers) - 1)  # in hours
    total_time = total_truck_time + total_drone_time + total_service_time
    
    return total_time, total_truck_time, total_drone_time, total_service_time, total_utilization, arrival_times


def calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time, current_time):
    """Calculate the delivery time using optimized cluster centers."""
    optimized_weights_reshaped = optimized_weights.reshape(-1, 2)
    optimized_centers = cluster_centers + optimized_weights_reshaped
    return calculate_clustering_delivery_time(optimized_centers, locations, labels, drone_speed, truck_speed, service_time,current_time)


# =========================
# Optimization Functions
# =========================


from pyswarm import pso

def optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, current_time):
    num_centers = len(cluster_centers)
    dim = num_centers * 2  # Each center has x and y weights

    # Objective Function with Penalty for Constraints
    def objective(weights):
        weights = np.array(weights).reshape(num_centers, 2)
        new_centers = cluster_centers + weights
        penalty = 0
        for idx, center in enumerate(new_centers):
            cluster_locations = locations[labels == idx]
            if not is_within_endurance(center, cluster_locations, drone_speed, service_time, DRONE_ENDURANCE):
                penalty += 1000  # Large penalty
        delivery_time = calculate_optimize_weights_delivery_time(weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time, current_time)[0]
        return delivery_time + penalty

    # Define Bounds
    lb = [-2] * dim
    ub = [2] * dim

    # Run PSO
    optimized_weights, fopt = pso(objective, lb, ub, swarmsize=50, maxiter=20, debug=False) 

    return optimized_weights.reshape(num_centers, 2)


""" def optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time,current_time):
    num_centers = len(cluster_centers)
    initial_weights = np.zeros((num_centers, 2))

    def penalty_function(weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
        penalty = 0
        new_centers = cluster_centers + weights.reshape(len(cluster_centers), 2)
        for idx, center in enumerate(new_centers):
            if not is_within_endurance(center, locations[labels == idx], drone_speed, service_time, DRONE_ENDURANCE):
                penalty += 1000
        return calculate_optimize_weights_delivery_time(weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time,current_time)[0] + penalty

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
        return initial_weights """

# =========================
# Performance Evaluation
# =========================

def evaluate_performance(n_iterations, current_time, drone_speed=40.0, truck_speed=60.0, service_time=5.0, disruption_probability=0.1, disruption_type='road_closure', num_robustness_simulations=100):
    """Evaluate the performance over multiple iterations."""
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

    # Array to store arrival times for the optimized model
    Optimize_arrival_times = []

    for i in range(n_iterations):
        print("Iteration:", i)
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

        cluster_time, cluster_truck_time, cluster_drone_time, cluster_service_time, drone_utilizations, _ = calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time,current_time)
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

        optimized_weights = optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, current_time)
        if optimized_weights is not None:
            optimize_time, optimize_truck_time, optimize_drone_time, optimize_service_time, drone_utilizations_opt, arrival_times_opt = calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time,current_time)
            Total_delivery_times_optimize.append(optimize_time)
            Total_travel_times_optimize.append(optimize_truck_time + optimize_drone_time)
            Total_truck_times_optimize.append(optimize_truck_time)
            Total_drone_times_optimize.append(optimize_drone_time)
            Total_service_times_optimize.append(optimize_service_time)
            Total_drone_utilization_optimize.append(drone_utilizations_opt)
            Optimize_arrival_times.append(arrival_times_opt)

            # Evaluate robustness for Optimized
            optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
            optimize_route = solve_tsp_with_gurobi(create_data_model(np.vstack([DEPOT, optimized_centers]), metric='cityblock'))
            mean_adjusted_time_optimize, std_adjusted_time_optimize, _ = evaluate_route_robustness(np.vstack([DEPOT, optimized_centers]), optimize_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_optimize.append((mean_adjusted_time_optimize, std_adjusted_time_optimize))

            optimized_centers_with_depot = np.vstack([DEPOT, optimized_centers])
            
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
                    Robustness_optimize=Total_robustness_optimize)  # Save optimized arrival times in CSV

    print_statistics(Total_delivery_times_tsp, Total_delivery_times_cluster, Total_delivery_times_optimize, Total_drone_times_cluster, Total_drone_times_optimize)
    print_robustness_statistics(Total_robustness_tsp, Total_robustness_cluster, Total_robustness_optimize)


# =========================
#  Data Saving
# =========================


def save_data_to_csv(filename, **data):
    # Define the path to the Hypothesis test folder
    folder = 'Prediction_model_effect/'
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        filename = filename.split('.')[0] + '_8.csv'
        while os.path.exists(os.path.join(folder, filename)):
            filename = filename.split('_')[0] + '_' + str(int(filename.split('_')[1].split('.')[0]) + 1) + '.csv'
        full_path = os.path.join(folder, filename)
    
    df = pd.DataFrame(data)
    df.to_csv(full_path, index=False)

# =========================
# Statistics
# =========================

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


def print_robustness_statistics(robustness_tsp, robustness_cluster, robustness_optimize):
    """Print robustness statistics."""
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


# =========================
# Main Execution
# =========================

def evaluate_performance_for_time_intervals(start, end, interval):
    """Evaluate performance for multiple time intervals."""
    for current_time in np.arange(start, end + interval, interval):
        print(f"\nEvaluating for current time: {current_time:02.0f}:00")
        main(current_time)

def main(current_time):

    evaluate_performance(
        n_iterations=5, 
        current_time=current_time,  # Pass current time to performance evaluation
        disruption_probability=0.1, 
        disruption_type='road_closure', 
        num_robustness_simulations=100
    )

if __name__ == "__main__":
    # Run the simulation for time intervals from 0 to 24 in steps of 2
    evaluate_performance_for_time_intervals(8, 22, 1)
    main()


