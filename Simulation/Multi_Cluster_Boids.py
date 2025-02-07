import pygame
import random
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, KDTree
import heapq
from cattle import Cattle
from wrangler import Wrangler
from astar import a_star, draw_path, smooth_path  # assumed to take (start, goal, obstacles)
from definitions import DBSCAN_MIN_SAMPLES, DBSCAN_EPS, SCREEN_WIDTH, SCREEN_HEIGHT, target_pos

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Wrangler Mustering Simulation with Clusters and Optimized A*")

obstacles = set()  # Example obstacles (if any)

# Initialize cattle
cattle_group = [
    Cattle(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50))
    for _ in range(110)
]


left_side_cattle = [c for c in cattle_group if c.pos[0] < 1000]
right_side_cattle = [c for c in cattle_group if c.pos[0] >= 1000]

# Initialize wrangler
wrangler = Wrangler()

# Function to compute the angle (in radians) between two vectors.
def angle_between(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    dot = np.dot(v1, v2)
    # Clamp dot value to avoid numerical issues
    dot = max(min(dot / (norm1 * norm2), 1.0), -1.0)
    return np.arccos(dot)

def process_clusters(cattle_positions, labels, screen):
    """
    Process clusters: if a cluster has 3 or more points, try drawing its convex hull.
    Otherwise, draw the centroid and/or a connecting line for 2-point clusters.
    """
    unique_labels = set(labels)
    clusters_info = []  # will hold (centroid, points, label)
    for label in unique_labels:
        # We include noise points (label == -1) as well.
        indices = np.where(labels == label)[0]
        cluster_points = cattle_positions[indices]
        centroid = np.mean(cluster_points, axis=0)
        clusters_info.append((centroid, cluster_points, label))
        
        if len(cluster_points) >= 3:
            # Try to compute and draw the convex hull.
            try:
                hull = ConvexHull(cluster_points, qhull_options='QJ')
                hull_points = cluster_points[hull.vertices]
                pygame.draw.polygon(screen, (200, 200, 255), hull_points, 2)
            except Exception as e:
                # If convex hull fails, simply draw the centroid.
                pygame.draw.circle(screen, (200, 200, 255), centroid.astype(int), 8)
        elif len(cluster_points) == 2:
            # For two points, draw a line connecting them.
            pygame.draw.line(screen, (200, 200, 255), cluster_points[0].astype(int), cluster_points[1].astype(int), 2)
            # Also draw the centroid.
            pygame.draw.circle(screen, (200, 200, 255), centroid.astype(int), 8)
        else:
            # Single point cluster: just draw a circle.
            pygame.draw.circle(screen, (200, 200, 255), centroid.astype(int), 8)
            
    return clusters_info


# Main loop
running = True
clock = pygame.time.Clock()
while running:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Remove cattle that are close to the target position
    cattle_group = [c for c in left_side_cattle if np.linalg.norm(c.pos - target_pos) > 60]

    # Update cattle positions; assume each cattle updates its own position
    clusters_for_move = []
    for cattle in cattle_group:
        cattle.move(wrangler, cattle_group, clusters_for_move)

    # Cluster cattle using DBSCAN if there are any cattle remaining
    if len(cattle_group) > 0:
        cattle_positions = np.array([c.pos for c in cattle_group])
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(cattle_positions)
        labels = dbscan.labels_

        clusters_info = process_clusters(cattle_positions, labels, screen)
        cluster_centroids = np.array([centroid for centroid, points, lab in clusters_info if lab != -999])
        cluster_labels = np.array([lab for centroid, points, lab in clusters_info if lab != -999])
        # -------------------------------
        # Optimized A* Evaluation:
        # For each cluster, run an A* search from the centroid to the target.
        # (This cost may be affected by obstacles if provided.)
        # Then, choose the cluster with the maximum A* path cost (furthest from the target).
        # -------------------------------
        cluster_costs = []
        for idx, centroid in enumerate(cluster_centroids):
            # Use the A* algorithm to plan a path from this centroid to the target.
            # We assume that a_star returns a list of points along the path.
            # (If obstacles are defined, a_star will plan around them.)
            path = a_star(centroid.astype(int), target_pos.astype(int), obstacles)
            path = smooth_path(path, smoothing_iterations=1)
            # Use the path length (or total cost if available) as the cost metric.
            if path and len(path) > 1:
                cost = 0
                for i in range(len(path)-1):
                    cost += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            else:
                cost = np.linalg.norm(centroid - target_pos)  # fallback to Euclidean distance
            cluster_costs.append(cost)

        # Identify the "furthest" cluster by the A* path cost.
        furthest_cluster_idx = np.argmax(cluster_costs)
        furthest_centroid = cluster_centroids[furthest_cluster_idx]

        # Highlight the furthest cluster's convex hull and centroid.
        furthest_cluster_points = cattle_positions[labels == cluster_labels[furthest_cluster_idx]]
        if len(furthest_cluster_points) >= 3:
            try:
                hull = ConvexHull(furthest_cluster_points)
                hull_points = furthest_cluster_points[hull.vertices]
                pygame.draw.polygon(screen, (255, 0, 0), hull_points, 2)  # Red hull for furthest cluster
            except Exception as e:
                print(f"Error drawing convex hull for furthest cluster: {e}")
        pygame.draw.circle(screen, (0, 255, 0), furthest_centroid.astype(int), 8)  # Green centroid

        candidate_idx = None
        best_candidate_score = float('inf')  # lower score is better

        for idx, centroid in enumerate(cluster_centroids):
            if idx == furthest_cluster_idx:
                continue

            # 1. Merging cost: distance between furthest cluster centroid and candidate cluster centroid.
            merge_dist = np.linalg.norm(centroid - furthest_centroid)
            
            # 2. Remaining cost: compute the A* path cost from the candidate's centroid to the target.
            path = a_star(centroid.astype(int), target_pos.astype(int), obstacles)
            path = smooth_path(path, smoothing_iterations=1)
            if path and len(path) > 1:
                remaining_cost = 0
                for i in range(len(path) - 1):
                    remaining_cost += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            else:
                remaining_cost = np.linalg.norm(centroid - target_pos)  # fallback to Euclidean distance
            
            # 3. Drone movement cost: distance from the drone's current position to the candidate centroid.
            # Assuming 'wrangler.pos' is the current position of the drone/wrangler.
            drone_cost = np.linalg.norm(wrangler.pos - centroid)
            
            # Combined score: lower scores represent less overall movement required.
            score = merge_dist + remaining_cost + drone_cost
            
            if score < best_candidate_score:
                best_candidate_score = score
                candidate_idx = idx



        neighbor_pos = None
        if candidate_idx is not None:
            neighbor_pos = cluster_centroids[candidate_idx]
            # Highlight the chosen neighbor cluster in yellow.
            pygame.draw.circle(screen, (255, 255, 0), neighbor_pos.astype(int), 8)

        # If a neighbor candidate was found, use A* to draw the merging path.
        if neighbor_pos is None:
            neighbor_pos = target_pos  # Fallback to target
            pygame.draw.circle(screen, (255, 255, 0), target_pos.astype(int), 20)

        path = a_star(furthest_centroid.astype(int), neighbor_pos.astype(int), obstacles)
        path = smooth_path(path, smoothing_iterations=1)
        if path:
            draw_path(screen, path)
            active_centroid = furthest_centroid
            # For demonstration, find a farthest point in the furthest cluster along the path direction.
            if len(path) > 1:
                smoothed_point = np.array(path[min(3, len(path)-1)])  # take a point further along the path
            else:
                smoothed_point = np.array(path[0])
            # Determine the farthest cattle within the furthest cluster relative to the merge direction.
            farthest_point_idx = np.argmax(np.linalg.norm(furthest_cluster_points - smoothed_point, axis=1))
            farthest_point = furthest_cluster_points[farthest_point_idx]
            # Find the actual Cattle object corresponding to the farthest point
            farthest_cattle = min(cattle_group, key=lambda c: np.linalg.norm(c.pos - farthest_point))


            pygame.draw.circle(screen, (255, 0, 255), farthest_point.astype(int), 10)
            # Update wrangler position based on the merge candidate and farthest point.
            wrangler.update_position(cattle_group, smoothed_point, farthest_cattle, active_centroid, screen)


    # Draw all cattle
    for cattle in cattle_group:
        cattle.draw(screen)

    # Draw the wrangler
    wrangler.draw(screen)

    # Draw the target position as a green dot
    pygame.draw.circle(screen, (0, 255, 0), target_pos.astype(int), 10)

    # (Optional) Draw a vertical divider if needed
    pygame.draw.line(screen, (0, 0, 0), (1000, 0), (1000, 750), 5)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
