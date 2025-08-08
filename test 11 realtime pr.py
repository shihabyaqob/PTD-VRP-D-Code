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
    with gp.Env(params=options) as env, gp.Model(env=env) as model:
        # Set Gurobi parameters for optimization
        model.setParam('Threads', 16)         # Use 16 threads for parallel processing
        model.setParam('MIPGap', 0.01)        # Set the MIP gap for optimization
        model.setParam('LogToConsole', 0)     # Suppress Gurobi log output

        # Define decision variables
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
        model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n)), name="outflow")
        model.addConstrs((gp.quicksum(x[j, i] for j in range(n) if j != i) == 1 for i in range(n)), name="inflow")
        
        # Subtour elimination constraints
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt

def get_truck_speed_at_time(delivery_time):
    # Load the dataset
    df = pd.read_csv('PS/TrafficTwoMonth.csv')

    # Convert 'Time' column to datetime format, spcifying the format 6:00:00 AM
    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')

    # Feature Engineering based on specific time
    # Extract hour and minute from the 'Time' column
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Time of Day'] = df['Hour'] + df['Minute'] / 60.0

    # Encode 'Day of the week' using LabelEncoder
    label_encoder = LabelEncoder()
    df['Day of the week'] = label_encoder.fit_transform(df['Day of the week'])

    # Create a binary feature for weekends
    df['Weekend'] = (df['Day of the week'] >= 5).astype(int)

    # Create interaction terms for vehicle counts
    df['Total Vehicle Count'] = df['CarCount'] + df['BikeCount'] + df['BusCount'] + df['TruckCount']
    df['Car_Bus_Count'] = df['CarCount'] * df['BusCount']
    df['Car_Truck_Count'] = df['CarCount'] * df['TruckCount']
    df['Bus_Truck_Count'] = df['BusCount'] * df['TruckCount']

    # Define speed ranges for different traffic situations
    speed_ranges = {
        'low': (80, 100),
        'normal': (50, 80),
        'high': (30, 50),
        'heavy': (20, 30)
    }

    # Vectorize the probabilistic speed mapping
    # Map 'Traffic Situation' to min_speed and max_speed
    df['min_speed'] = df['Traffic Situation'].map(lambda x: speed_ranges[x][0])
    df['max_speed'] = df['Traffic Situation'].map(lambda x: speed_ranges[x][1])

    # Compute 'bias_factor' based on 'Time of Day'
    rush_hour = ((df['Time of Day'] >= 7) & (df['Time of Day'] <= 9)) | ((df['Time of Day'] >= 17) & (df['Time of Day'] <= 19))
    df['bias_factor'] = np.where(rush_hour, 0.7, 0.3)

    # Adjust 'min_speed' and 'max_speed' based on 'Total Vehicle Count'
    df['adjusted_min_speed'] = df['min_speed'] + np.where(df['Total Vehicle Count'] < 20, 5, 0)
    df['adjusted_max_speed'] = df['max_speed'] - np.where(df['Total Vehicle Count'] > 50, 5, 0)

    # Ensure 'adjusted_min_speed' does not exceed 'adjusted_max_speed'
    df['adjusted_min_speed'] = np.minimum(df['adjusted_min_speed'], df['adjusted_max_speed'])

    # Compute the lower bound for uniform distribution
    df['lower_bound'] = df['adjusted_min_speed'] + df['bias_factor'] * (df['adjusted_max_speed'] - df['adjusted_min_speed'])

    # Generate 'Probabilistic Speed'
    np.random.seed()  
    df['Probabilistic Speed'] = np.random.uniform(df['lower_bound'], df['adjusted_max_speed'])

    # Use the new probabilistic speed as the target variable
    y = df['Probabilistic Speed']

    # Select relevant features for prediction (including 'Time of Day')
    X = df[['Time of Day', 'Day of the week', 'Weekend', 'Total Vehicle Count', 
            'Car_Bus_Count', 'Car_Truck_Count', 'Bus_Truck_Count']]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBRegressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_regressor.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = xgb_regressor.predict(X_test)

    # Predict the truck speed at the given delivery time
    x_pred = np.array([[delivery_time, 1, 0, 50, 10, 15, 20]])
    prediction = xgb_regressor.predict(x_pred)

    return prediction[0]


def calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, max_drones_per_cluster=N_DRONES):
    """Calculate the total delivery time based on clustering, optimized for time-dependent speeds."""
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    tsp_route = solve_tsp_or_tools(create_data_model(cluster_centers_with_depot, metric='cityblock'))
    if tsp_route is None:
        return None, None

    truck_distances = cdist(cluster_centers_with_depot[tsp_route[:-1]], cluster_centers_with_depot[tsp_route[1:]], metric='cityblock').diagonal()
    
    total_truck_time = 0
    total_drone_time = 0
    total_utilization = 0
    cluster_count = 0  # To count the number of clusters processed
    
    current_time = 16.0  # Start simulation at 8 AM (or adjust as needed)
    arrival_times = []  # List to store the arrival times at each cluster center

    for i, center_idx in enumerate(tsp_route[1:-1]):
        travel_distance = truck_distances[i]
        remaining_distance = travel_distance
        segment_travel_time = 0

        # Continue until the truck reaches the destination, crossing time windows if necessary
        while remaining_distance > 0:
            truck_speed = get_truck_speed_at_time(current_time)  # Get the speed for the current time
            print("Truck Speed at Time {:.2f}: {:.2f} km/h".format(current_time, truck_speed))
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
        round_trip_times = 2 * delivery_times  # Out and back for each delivery point

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


def calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
    """Calculate the delivery time using optimized cluster centers."""
    optimized_weights_reshaped = optimized_weights.reshape(-1, 2)
    optimized_centers = cluster_centers + optimized_weights_reshaped
    return calculate_clustering_delivery_time(optimized_centers, locations, labels, drone_speed, truck_speed, service_time)


# =========================
# Optimization Functions
# =========================


def optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
    """Optimize the cluster centers to minimize delivery time."""
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


# =========================
# Performance Evaluation
# =========================

def evaluate_performance(n_iterations, drone_speed=40.0, truck_speed=60.0, service_time=5.0, disruption_probability=0.1, disruption_type='road_closure', num_robustness_simulations=100):
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

        tsp_route = solve_tsp_or_tools(create_data_model(locations, metric='cityblock'))
        if tsp_route:
            tsp_time, tsp_truck_time, tsp_service_time = calculate_tsp_delivery_time(tsp_route, locations, truck_speed, service_time)
            Total_delivery_times_tsp.append(tsp_time)
            Total_travel_times_tsp.append(tsp_truck_time)
            Total_truck_times_tsp.append(tsp_truck_time)
            Total_service_times_tsp.append(tsp_service_time)

            # Evaluate robustness for TSP
            mean_adjusted_time_tsp, std_adjusted_time_tsp, _ = evaluate_route_robustness(locations, tsp_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_tsp.append((mean_adjusted_time_tsp, std_adjusted_time_tsp))

        cluster_time, cluster_truck_time, cluster_drone_time, cluster_service_time, drone_utilizations, _ = calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if cluster_time:
            Total_delivery_times_cluster.append(cluster_time)
            Total_travel_times_cluster.append(cluster_truck_time + cluster_drone_time)
            Total_truck_times_cluster.append(cluster_truck_time)
            Total_drone_times_cluster.append(cluster_drone_time)
            Total_service_times_cluster.append(cluster_service_time)
            Total_drone_utilization_cluster.append(drone_utilizations)

            # Evaluate robustness for Clustering
            cluster_route = solve_tsp_or_tools(create_data_model(np.vstack([DEPOT, cluster_centers]), metric='cityblock'))
            mean_adjusted_time_cluster, std_adjusted_time_cluster, _ = evaluate_route_robustness(np.vstack([DEPOT, cluster_centers]), cluster_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_cluster.append((mean_adjusted_time_cluster, std_adjusted_time_cluster))

        optimized_weights = optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if optimized_weights is not None:
            optimize_time, optimize_truck_time, optimize_drone_time, optimize_service_time, drone_utilizations_opt, arrival_times_opt = calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
            Total_delivery_times_optimize.append(optimize_time)
            Total_travel_times_optimize.append(optimize_truck_time + optimize_drone_time)
            Total_truck_times_optimize.append(optimize_truck_time)
            Total_drone_times_optimize.append(optimize_drone_time)
            Total_service_times_optimize.append(optimize_service_time)
            Total_drone_utilization_optimize.append(drone_utilizations_opt)
            Optimize_arrival_times.append(arrival_times_opt)

            # Evaluate robustness for Optimized
            optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
            optimize_route = solve_tsp_or_tools(create_data_model(np.vstack([DEPOT, optimized_centers]), metric='cityblock'))
            mean_adjusted_time_optimize, std_adjusted_time_optimize, _ = evaluate_route_robustness(np.vstack([DEPOT, optimized_centers]), optimize_route, truck_speed, service_time, num_robustness_simulations, disruption_probability, disruption_type)
            Total_robustness_optimize.append((mean_adjusted_time_optimize, std_adjusted_time_optimize))

            optimized_centers_with_depot = np.vstack([DEPOT, optimized_centers])
            save_iter_data_to_csv(i, locations, labels, optimized_centers_with_depot, optimize_route, arrival_times_opt)
            
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

def save_iter_data_to_csv(iteration, locations, labels, optimized_centers, opt_route, arrival_times, csv_file='all_iterations_data.csv'):
    """Save iteration-specific data to a CSV file."""
    # Ensure that the iteration is a single integer and not an array
    if isinstance(iteration, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected an integer for iteration, but got {iteration}")
    
    # Create the folder if it doesn't exist
    folder_name = 'Iteration_Data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Convert lists to JSON format so they can be saved as arrays in the CSV
    locations_str = json.dumps(locations.tolist())  # Convert locations array to JSON string
    labels_str = json.dumps(labels.tolist())  # Convert labels array to JSON string
    centers_str = json.dumps(optimized_centers.tolist())  # Convert cluster centers array to JSON string
    tsp_route_str = json.dumps(opt_route)  # Convert TSP route to JSON string
    arrival_times_str = json.dumps(arrival_times)  # Convert arrival times to JSON string
    
    # Prepare a DataFrame to store the data for this iteration
    iteration_data = pd.DataFrame({
        'iteration': [iteration],  # Ensure this is a single integer
        'locations': [locations_str],  # Store locations as a JSON string
        'labels': [labels_str],  # Store labels as a JSON string
        'cluster_centers': [centers_str],  # Store cluster centers as a JSON string
        'opt_route': [tsp_route_str],  # Store TSP route as a JSON string
        'arrival_times': [arrival_times_str]  # Store arrival times as a JSON string
    })
    
    # Save the DataFrame to the CSV file
    csv_path = f'{folder_name}/{csv_file}'
    if iteration == 0:
        iteration_data.to_csv(csv_path, index=False, mode='w')  # 'w' for write mode (overwrite)
    else:
        iteration_data.to_csv(csv_path, index=False, mode='a', header=False)  # 'a' for append, no header for subsequent iterations


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

def save_data_to_csv(filename, **data):
    """Save data to a CSV file."""
    if filename == 'iter_opt_arrival_times.csv':
        filename = 'Route/' + filename
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


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

def main():
    """Main function to run the simulation and evaluate performance."""
    locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
    labels, cluster_centers = cluster_locations(locations)
    initial_data = create_data_model(locations, metric='cityblock')
    initial_tsp_route = solve_tsp_or_tools(initial_data)
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    data = create_data_model(cluster_centers_with_depot, metric='cityblock')
    tsp_route = solve_tsp_or_tools(data)

    if tsp_route and initial_tsp_route:
        optimized_weights = optimize_weights(cluster_centers, locations, labels, DRONE_SPEED, DEFAULT_TRUCK_SPEED, SERVICE_TIME)
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
