import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.optimize import minimize
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Constants
AREA_SIZE = 10.0  # km
N_DELIVERY_LOCATIONS = 30
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

def solve_tsp(data):
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


def calculate_tsp_delivery_time(route, locations, truck_speed, service_time):
    truck_distances = cdist(locations[route[:-1]], locations[route[1:]], metric='cityblock').diagonal()
    truck_time = np.sum(truck_distances) / truck_speed  # this is in hours
    print ("Sum of Truck Distances: ", np.sum(truck_distances))
    print ("Truck Time: ", truck_time)
    service_time_hours = service_time / 60  # converting minutes to hours
    total_time = truck_time + service_time_hours * (len(route) - 1)  # service at every stop except depot return

    return total_time, truck_time

def calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time, max_drones_per_cluster=N_DRONES):
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    
    tsp_route = solve_tsp(create_data_model(cluster_centers_with_depot, metric='cityblock'))
    if tsp_route is None:
        return None, None

    truck_distances = cdist(cluster_centers_with_depot[tsp_route[:-1]], cluster_centers_with_depot[tsp_route[1:]], metric='cityblock').diagonal()
    truck_time = np.sum(truck_distances) / truck_speed  # in hours

    total_drone_time = 0
    
    for center_idx in tsp_route[1:-1]:
        cluster_locs = locations[labels == (center_idx - 1)]
        if cluster_locs.size == 0:
            continue

        distances = np.linalg.norm(cluster_locs - cluster_centers_with_depot[center_idx], axis=1)  # distance to each delivery point
        delivery_times = distances / drone_speed  # time to each delivery point
        service_times = np.full_like(delivery_times, service_time / 60)  # convert service time to hours
        round_trip_times = 2 * delivery_times + service_times  # out and back for each delivery point

        drone_end_times = np.zeros(max_drones_per_cluster)  # end times for each drone's deliveries
        for trip_time in round_trip_times:
            earliest_drone = np.argmin(drone_end_times)
            drone_end_times[earliest_drone] += trip_time
        
        total_drone_time += np.max(drone_end_times)

    total_time = truck_time + total_drone_time
    return total_time, truck_time, total_drone_time

def calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
    optimized_weights_reshaped = optimized_weights.reshape(-1, 2)
    optimized_centers = cluster_centers + optimized_weights_reshaped
    return calculate_clustering_delivery_time(optimized_centers, locations, labels, drone_speed, truck_speed, service_time)


def optimize_optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time):
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


def evaluate_performance(n_iterations, drone_speed=40.0, truck_speed=60.0, service_time=5.0):
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


    
    for _ in range(n_iterations):
        print("Iteration:", _)
        locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
        labels, cluster_centers = cluster_locations(locations)

        tsp_route = solve_tsp(create_data_model(locations, metric='cityblock'))
        if tsp_route:
            tsp_time, tsp_truck_time = calculate_tsp_delivery_time(tsp_route, locations, truck_speed, service_time)
            Total_delivery_times_tsp.append(tsp_time)
            Total_travel_times_tsp.append(tsp_truck_time)
            Total_truck_times_tsp.append(tsp_truck_time)

        cluster_time, cluster_truck_time, cluster_drone_time= calculate_clustering_delivery_time(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if cluster_time:
            Total_delivery_times_cluster.append(cluster_time)
            Total_travel_times_cluster.append(cluster_truck_time+cluster_drone_time)
            Total_truck_times_cluster.append(cluster_truck_time)
            Total_drone_times_cluster.append(cluster_drone_time)


        optimized_weights = optimize_optimize_weights(cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
        if optimized_weights is not None:
            optimize_time, optimize_truck_time,optimize_drone_time = calculate_optimize_weights_delivery_time(optimized_weights, cluster_centers, locations, labels, drone_speed, truck_speed, service_time)
            Total_delivery_times_optimize.append(optimize_time)
            Total_travel_times_optimize.append(optimize_truck_time+optimize_drone_time)
            Total_truck_times_optimize.append(optimize_truck_time)
            Total_drone_times_optimize.append(optimize_drone_time)

    plot_delivery_times(n_iterations, Total_delivery_times_tsp, Total_delivery_times_cluster, Total_delivery_times_optimize)
    plot_travel_times(n_iterations, Total_travel_times_tsp, Total_travel_times_cluster, Total_travel_times_optimize)
    plot_truck_times(n_iterations, Total_truck_times_tsp, Total_truck_times_cluster, Total_truck_times_optimize)
    plot_drone_times(n_iterations, Total_drone_times_cluster, Total_drone_times_optimize)
    print_statistics(Total_delivery_times_tsp, Total_delivery_times_cluster, Total_delivery_times_optimize, Total_drone_times_cluster, Total_drone_times_optimize)
    
def print_statistics(total_tsp, total_cluster, total_optimize, total_drone_times_cluster, total_drone_times_optimize):
    print("Average Total Delivery Time (TSP):", np.mean(total_tsp))
    print("Average Total Delivery Time (Clustering):", np.mean(total_cluster))
    print("Average Total Delivery Time (Optimize Weights):", np.mean(total_optimize))
    print("Average Total Drone Time (Clustering):", np.mean(total_drone_times_cluster))
    print("Average Total Drone Time (Optimize Weights):", np.mean(total_drone_times_cluster))
    improvement_cluster = ((np.mean(total_tsp) - np.mean(total_cluster)) / np.mean(total_tsp)) * 100
    improvement_optimize = ((np.mean(total_cluster) - np.mean(total_optimize)) / np.mean(total_cluster)) * 100
    print("Improvement from TSP to Clustering:", improvement_cluster, "%")
    print("Improvement from Clustering to Optimize Weights:", improvement_optimize, "%")

def plot_routes(locations, labels, cluster_centers, tsp_route, initial_tsp_route=None, optimized_centers=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')

    if initial_tsp_route is not None:
        initial_route = np.array([locations[i] for i in initial_tsp_route])
        plt.plot(initial_route[:, 0], initial_route[:, 1], 'r-', label='Initial TSP Route')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='green', marker='x', label='Cluster Centers')

    if tsp_route is not None:
        tsp_route_coords = np.array([cluster_centers[i] for i in tsp_route])
        plt.plot(tsp_route_coords[:, 0], tsp_route_coords[:, 1], 'g--', label='Cluster Centers TSP Route')

    if optimized_centers is not None:
        plt.scatter(optimized_centers[:, 0], optimized_centers[:, 1], c='purple', marker='s', label='Optimized Centers')
        if tsp_route is not None:
            optimized_route_coords = np.array([optimized_centers[i] for i in tsp_route])
            plt.plot(optimized_route_coords[:, 0], optimized_route_coords[:, 1], 'm--', label='Optimized Centers TSP Route')
    
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Routes and Locations')
    plt.grid(True)
    plt.show()

def plot_TSP_inital_route(locations, initial_tsp_route, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, cmap='tab10', label='Delivery Locations')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', marker='*', s=200, label='Depot')
    initial_route_coords = np.array([locations[i] for i in initial_tsp_route])
    plt.plot(initial_route_coords[:, 0], initial_route_coords[:, 1], 'r-', label='Initial TSP Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
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
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
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
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
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
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
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

def main():
    locations = generate_random_locations(N_DELIVERY_LOCATIONS, DEPOT)
    labels, cluster_centers = cluster_locations(locations)
    initial_data = create_data_model(locations, metric='cityblock')
    initial_tsp_route = solve_tsp(initial_data)
    cluster_centers_with_depot = np.vstack([DEPOT, cluster_centers])
    data = create_data_model(cluster_centers_with_depot, metric='cityblock')
    tsp_route = solve_tsp(data)

    if tsp_route and initial_tsp_route:
        optimized_weights = optimize_optimize_weights(
            cluster_centers, locations, labels, DRONE_SPEED, TRUCK_SPEED, SERVICE_TIME
        )
        if optimized_weights is not None:
            optimized_centers = cluster_centers + optimized_weights.reshape(-1, 2)
            optimized_centers_with_depot = np.vstack([DEPOT, optimized_centers])
            
            """ 
            plot_routes(locations, labels, cluster_centers_with_depot, tsp_route, initial_tsp_route=initial_tsp_route, optimized_centers=optimized_centers_with_depot)
            plot_TSP_inital_route(locations, initial_tsp_route, labels)
            plot_cluster_route(locations, cluster_centers_with_depot, tsp_route, labels)
            plot_optimize_route(locations, optimized_centers_with_depot, tsp_route, labels)
            plot_cluster_optimize_route(locations, cluster_centers_with_depot, optimized_centers_with_depot, tsp_route, labels) """
            
            evaluate_performance(n_iterations=30)
        else:
            print("Optimization failed.")
    else:
        print("TSP solution not found.")

if __name__ == "__main__":
    main()

