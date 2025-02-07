import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
from definitions import SEPARATION_RADIUS, SEPARATION_WEIGHT, ALIGNMENT_RADIUS, ALIGNMENT_WEIGHT, COHESION_RADIUS, COHESION_WEIGHT, MAX_SPEED, MAX_FORCE, DBSCAN_MIN_SAMPLES, DBSCAN_EPS, SCREEN_HEIGHT, SCREEN_WIDTH, target_pos


def add_goal_as_centroid(centroids, goal_position):
    updated_centroids = np.vstack([centroids, goal_position])
    return updated_centroids

# Function to compute MST using NetworkX
def compute_mst(positions):
    G = nx.Graph()
    updated_centroids = add_goal_as_centroid(positions, target_pos)
    distance_matrix_values = distance_matrix(updated_centroids, updated_centroids)
    num_nodes = len(positions)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix_values[i][j])

    mst = nx.minimum_spanning_tree(G)
    return mst