from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

# =========================
# Constants
# =========================
AREA_SIZE = 10.0  # km
DEPOT = (0, 0)
DRONE_ENDURANCE = 20  # minutes
DRONE_SPEED = 40.0  # km/h
SERVICE_TIME = 5.0  # in minutes
np.random.seed(42)  # Set a specific seed for reproducibility

# =========================
# Utility Functions
# =========================

def generate_random_locations(n_locations, depot, range_km=AREA_SIZE):
    """Generate random delivery locations within a specified range."""
    locations = np.random.uniform(-range_km, range_km, size=(n_locations, 2))
    return np.vstack([depot, locations])

def is_within_endurance_vectorized(cluster_centers, locations, labels, drone_speed, service_time, endurance):
    """Vectorized check to determine if all locations in each cluster are within the drone's endurance."""
    distances = np.linalg.norm(locations - cluster_centers[labels], axis=1)
    round_trip_times = 2 * (distances / drone_speed) * 60
    total_times = round_trip_times + service_time

    n_clusters = cluster_centers.shape[0]
    within_endurance = np.array([np.all(total_times[labels == i] <= endurance) for i in range(n_clusters)])

    return within_endurance

def cluster_locations(locations, max_clusters=100):
    """Cluster locations using KMeans until all clusters meet the endurance constraint."""
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(locations)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        within_endurance = is_within_endurance_vectorized(
            cluster_centers=centers,
            locations=locations,
            labels=labels,
            drone_speed=DRONE_SPEED,
            service_time=SERVICE_TIME,
            endurance=DRONE_ENDURANCE
        )

        if np.all(within_endurance):
            cluster_sizes = np.bincount(labels, minlength=n_clusters)
            print(f"Valid clustering found with {n_clusters} clusters.")
            print(f"Cluster sizes: {cluster_sizes.tolist()}")
            return labels, centers

    raise ValueError("Failed to find a valid clustering configuration within the maximum cluster limit.")

def single_iteration(n_delivery_locations, iteration_number):
    """Perform a single iteration of adaptive clustering."""
    locations = generate_random_locations(n_delivery_locations, DEPOT)
    delivery_locations = locations[1:]  # Exclude depot for clustering

    try:
        labels, cluster_centers = cluster_locations(delivery_locations)
        num_clusters = len(np.unique(labels))
        cluster_sizes = np.bincount(labels)
        cluster_sizes_str = ','.join(map(str, cluster_sizes.tolist()))

        return {
            'Iteration': iteration_number,
            'Num_Delivery_Locations': n_delivery_locations,
            'Num_Clusters': num_clusters,
            'Cluster_Sizes': cluster_sizes_str
        }
    except ValueError as ve:
        print(f"Iteration {iteration_number} for {n_delivery_locations} locations: {ve}")
        return {
            'Iteration': iteration_number,
            'Num_Delivery_Locations': n_delivery_locations,
            'Num_Clusters': np.nan,  # Use NaN for failed attempts
            'Cluster_Sizes': None
        }

def run_adaptive_clustering_simulation(n_delivery_locations, n_iterations=100):
    """Run clustering simulations for a given number of delivery locations."""
    args = [(n_delivery_locations, i + 1) for i in range(n_iterations)]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.starmap(single_iteration, args), total=n_iterations, desc=f"Simulating {n_delivery_locations} locations"))

    return results

def save_simulation_results_to_csv(results, n_delivery_locations):
    """Save simulation results to CSV."""
    folder = 'adaptive_clustering'
    os.makedirs(folder, exist_ok=True)

    filename = f'Adaptive_Clustering_Results_{n_delivery_locations}.csv'
    full_path = os.path.join(folder, filename)

    df = pd.DataFrame(results)
    df.to_csv(full_path, index=False)
    print(f"Saved results for {n_delivery_locations} locations to {full_path}")

def main():
    """Main function to execute the clustering simulations."""
    delivery_location_values = range(10, 101, 10)  # Delivery locations: 10, 20, ..., 100
    n_iterations = 100  # Increase iterations for better statistics

    all_results = []

    for n_delivery_locations in delivery_location_values:
        print(f"\nRunning simulations for {n_delivery_locations} delivery locations.")
        results = run_adaptive_clustering_simulation(n_delivery_locations, n_iterations)
        all_results.extend(results)
        save_simulation_results_to_csv(results, n_delivery_locations)

    df_results = pd.DataFrame(all_results)

    # Group by delivery locations and calculate summary statistics
    summary = df_results.groupby('Num_Delivery_Locations')['Num_Clusters'].describe()

    # Handle NaN values correctly for better reporting
    summary.fillna('N/A', inplace=True)

    # Save the summary to CSV
    summary_path = 'adaptive_clustering/summary.csv'
    summary.to_csv(summary_path)
    print(f"\nSummary Statistics saved to {summary_path}")
    print(summary)

if __name__ == "__main__":
    main()
