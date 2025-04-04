#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import math
from std_msgs.msg import ColorRGBA
import time

class EnhancedCattleVisualizerNode(Node):
    """
    ROS2 node for visualizing cattle in RViz with optimal mustering path planning,
    cluster prioritization, and push direction visualization.
    """
    
    def __init__(self):
        super().__init__('enhanced_cattle_visualizer_node')
        
        # Parameters
        self.num_cows = 12
        self.update_rate = 10  # Hz
        
        # DBSCAN parameters
        self.eps = 3.0        # Maximum distance between two samples to be in same cluster
        self.min_samples = 2  # Minimum samples in a cluster
        
        # Herding parameters
        self.min_approach_distance = 3.0  # Minimum distance to approach a cluster
        self.approach_angle_range = math.pi/4  # Range of approach angles (45 degrees)
        
        # Path planning update parameters
        self.weight_change_threshold = 0.5  # Threshold for significant weight change (20%)
        self.previous_cluster_weights = {}  # Store previous weights for comparison
        self.path_last_updated = time.time()  # Timestamp of last update
        self.min_update_interval = 1.0  # Minimum time between updates (seconds)
        self.force_update_counter = 0  # Counter to force update after several cycles
        self.force_update_threshold = 300  # Update path every N cycles regardless of weight changes

        # State tracking
        self.cattle_positions = {}
        self.wrangler_position = np.array([0.0, 0.0, 0.0])
        self.previous_wrangler_positions = []
        self.max_path_length = 100
        self.data_received = False  # Flag to track if we've received any data
        
        # Mustering planning
        self.clusters = []  # Will store all cluster data
        self.prioritized_clusters = []  # Ordered clusters based on priority

        self.current_target_cluster = None  # Currently targeted cluster
        self.optimal_path = []  # Sequence of positions to follow
        self.path_segments = {}  # Stores path segments
        
        # Mock data for initial visualization (will be replaced with real data)
        self.initialize_mock_data()
        
        # Track active cluster markers for cleanup
        self.current_cluster_count = 0
        self.previous_cluster_markers = {
            "cluster_hull": set(),
            "cluster_hull_fill": set(),
            "cluster_text": set(),
            "cluster_line": set(),
            "optimal_path": set(),
            "approach_sector": set(),
            "target_positions": set()  # New entry for target markers
        }
        
        # Create publishers for visualization
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            '/visualization/markers', 
            10
        )
        
        self.goal_pub = self.create_publisher(
            PointStamped,
            '/visualization/goal',
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/visualization/path',
            10
        )
        
        self.optimal_path_pub = self.create_publisher(
            Path,
            '/visualization/optimal_path',
            10
        )
        
        # New publishers for target positions
        self.target_pub = self.create_publisher(
            PointStamped,
            '/visualization/current_target',
            10
        )
        
        self.next_target_pub = self.create_publisher(
            PointStamped,
            '/visualization/next_target',
            10
        )
        
        # Subscribe to cattle odometry
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
            
            # Also try alternative topic format
            self.create_subscription(
                Odometry,
                f'/cow{i}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
        
        # Subscribe to wrangler/drone odometry (try multiple potential topic names)
        self.create_subscription(
            Odometry,
            '/drone/odom',
            self.wrangler_callback,
            10
        )
        
        # Try alternative topic names for the wrangler
        self.create_subscription(
            Odometry,
            '/wrangler/odom',
            self.wrangler_callback,
            10
        )
        
        self.create_subscription(
            Odometry,
            '/robot/odom',
            self.wrangler_callback,
            10
        )
        
        # Timer for visualization updates
        self.update_timer = self.create_timer(1.0/self.update_rate, self.update_visualization)
        
        # Set goal position
        self.goal_position = np.array([7.0, 7.0, 0.0])
        
        # Publish initial goal visualization
        self.publish_goal()
        
        # Store cluster data for debugging
        self.cluster_data = {}
        
        # Special timer for debug info
        self.debug_timer = self.create_timer(5.0, self.print_debug_info)
        
        self.get_logger().info('Enhanced Cattle Visualizer node initialized with adaptive path planning')
        
        # Immediately publish initial visualization
        self.update_visualization()
    
    def initialize_mock_data(self):
        """
        Initialize some mock data for visualization until real data arrives
        """
        self.get_logger().info('Initializing mock cattle data for visualization')
        
        # Create a grid of cattle
        grid_size = 3  # 3x3 grid
        spacing = 2.0  # 2 meters between cattle
        
        for i in range(grid_size):
            for j in range(grid_size):
                cow_id = f'cow{i*grid_size + j + 1}'
                # Position cattle in a grid pattern with small random offsets
                x = -5.0 + i * spacing + np.random.uniform(-0.3, 0.3)
                y = -5.0 + j * spacing + np.random.uniform(-0.3, 0.3)
                z = 0.0
                
                self.cattle_positions[cow_id] = np.array([x, y, z])
        
        # Add a few more scattered cattle
        for i in range(grid_size*grid_size + 1, self.num_cows + 1):
            cow_id = f'cow{i}'
            # Random positions for remaining cattle
            x = np.random.uniform(-8.0, 0.0)
            y = np.random.uniform(-8.0, 0.0)
            z = 0.0
            
            self.cattle_positions[cow_id] = np.array([x, y, z])
                
        # Set wrangler initial position
        self.wrangler_position = np.array([-10.0, -10.0, 2.0])
        
        # Create initial wrangler path point
        timestamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = timestamp
        pose.pose.position.x = self.wrangler_position[0]
        pose.pose.position.y = self.wrangler_position[1]
        pose.pose.position.z = self.wrangler_position[2]
        pose.pose.orientation.w = 1.0
        
        self.previous_wrangler_positions.append(pose)
    
    def print_debug_info(self):
        """
        Print debug information about the current state
        """
        self.get_logger().info(f"Debug Info - Cattle count: {len(self.cattle_positions)}, Cluster count: {len(self.clusters)}")
        self.get_logger().info(f"Wrangler position: {self.wrangler_position}")
        self.get_logger().info(f"Data received flag: {self.data_received}")
        
        # Check if we've received any cattle positions
        if not self.cattle_positions:
            self.get_logger().warn("No cattle positions received - check topic names and data sources")
        
        # Force visibility of markers
        self.publish_goal()
        self.publish_path()
        self.publish_optimal_path()
        self.publish_target_positions()  # New method call
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions from odometry
        """
        pos = msg.pose.pose.position
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        self.data_received = True
        self.get_logger().debug(f"Received data for {cattle_id}: [{pos.x}, {pos.y}, {pos.z}]")
    
    def wrangler_callback(self, msg):
        """
        Update wrangler position from odometry and store for path visualization
        """
        pos = msg.pose.pose.position
        self.wrangler_position = np.array([pos.x, pos.y, pos.z])
        self.data_received = True
        self.get_logger().debug(f"Received wrangler position: [{pos.x}, {pos.y}, {pos.z}]")
        
        # Add to path history
        timestamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = timestamp
        pose.pose.position.x = pos.x
        pose.pose.position.y = pos.y
        pose.pose.position.z = pos.z
        pose.pose.orientation = msg.pose.pose.orientation
        
        self.previous_wrangler_positions.append(pose)
        
        # Keep path length within limits
        if len(self.previous_wrangler_positions) > self.max_path_length:
            self.previous_wrangler_positions = self.previous_wrangler_positions[-self.max_path_length:]
    
    def publish_goal(self):
        """
        Publish goal position as a point for visualization
        """
        goal_msg = PointStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.point.x = self.goal_position[0]
        goal_msg.point.y = self.goal_position[1]
        goal_msg.point.z = self.goal_position[2]
        
        self.goal_pub.publish(goal_msg)
    
    def publish_target_positions(self):
        """
        Publish the positions of the current target cluster and the next target
        """
        # Only proceed if we have prioritized clusters and path segments
        if not self.prioritized_clusters or not self.path_segments:
            return
            
        # The current target is the highest priority cluster
        current_target_idx = self.prioritized_clusters[0][0]
        current_target_cluster = self.clusters[current_target_idx]
        current_target_position = current_target_cluster['centroid']
        
        # Publish current target position
        target_msg = PointStamped()
        target_msg.header.frame_id = "map"
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.point.x = current_target_position[0]
        target_msg.point.y = current_target_position[1]
        target_msg.point.z = current_target_position[2]
        
        self.target_pub.publish(target_msg)
        
        # Find the next target (the cluster the current target is being pushed towards)
        if current_target_idx in self.path_segments:
            next_target_idx = self.path_segments[current_target_idx]['merge_with']
            
            # If there's a next target cluster, publish its position
            if next_target_idx is not None:
                next_target_cluster = self.clusters[next_target_idx]
                next_target_position = next_target_cluster['centroid']
                
                next_target_msg = PointStamped()
                next_target_msg.header.frame_id = "map"
                next_target_msg.header.stamp = self.get_clock().now().to_msg()
                next_target_msg.point.x = next_target_position[0]
                next_target_msg.point.y = next_target_position[1]
                next_target_msg.point.z = next_target_position[2]
                
                self.next_target_pub.publish(next_target_msg)
            else:
                # If going directly to goal, publish goal as next target
                next_target_msg = PointStamped()
                next_target_msg.header.frame_id = "map"
                next_target_msg.header.stamp = self.get_clock().now().to_msg()
                next_target_msg.point.x = self.goal_position[0]
                next_target_msg.point.y = self.goal_position[1]
                next_target_msg.point.z = self.goal_position[2]
                
                self.next_target_pub.publish(next_target_msg)
    
    def publish_path(self):
        """
        Publish the historical path of the wrangler
        """
        historical_path = Path()
        historical_path.header.frame_id = "map"
        historical_path.header.stamp = self.get_clock().now().to_msg()
        historical_path.poses = self.previous_wrangler_positions
        
        self.path_pub.publish(historical_path)
    
    def check_significant_weight_changes(self, new_prioritized_clusters):
        """
        Check if there are significant changes in cluster weights that warrant path re-planning
        
        Returns:
        - True if significant changes detected, False otherwise
        """
        # Always update if we haven't created path segments yet
        if not hasattr(self, 'path_segments') or not self.path_segments:
            self.get_logger().debug("Initial path planning needed")
            return True
            
        # Check if clusters have changed (different count)
        if abs(len(new_prioritized_clusters) - len(self.prioritized_clusters))> 0:
            self.get_logger().info(f"Cluster count changed: {len(self.prioritized_clusters)} -> {len(new_prioritized_clusters)}")
            return True
            
        # Force update after certain number of cycles
        self.force_update_counter += 1
        if self.force_update_counter >= self.force_update_threshold:
            self.force_update_counter = 0
            self.get_logger().debug("Forced path update due to cycle threshold")
            return False
        
        # Check for significant weight changes
        significant_change = False
        
        for (new_idx, new_weight), (old_idx, old_weight) in zip(new_prioritized_clusters, self.prioritized_clusters):
            # Compare cluster indices (if different clusters have different priority now)
            if new_idx != old_idx:
                self.get_logger().info(f"Cluster priority order changed: {old_idx} -> {new_idx}")
                significant_change = False
                break
                
            # Check for significant change in weight value
            weight_change_ratio = abs(new_weight - old_weight) / (abs(old_weight) + 1e-6)
            if weight_change_ratio > self.weight_change_threshold:
                self.get_logger().debug(f"Significant weight change for cluster {new_idx}: {old_weight:.2f} -> {new_weight:.2f} (change: {weight_change_ratio:.2f})")
                significant_change = False
                break
        
        # Check minimum time interval between updates
        current_time = time.time()
        time_since_last_update = current_time - self.path_last_updated
        
        if significant_change and time_since_last_update < self.min_update_interval:
            self.get_logger().debug(f"Delaying update: {time_since_last_update:.2f}s since last update")
            return False
            
        if significant_change:
            self.path_last_updated = current_time
            
        return significant_change

    def calculate_cluster_metrics(self, clusters):
        """
        Calculate metrics for each cluster to prioritize them for mustering
        
        Returns:
        - A list of (cluster_idx, priority_score) tuples sorted by priority
        """
        priorities = []
        
        for idx, cluster in enumerate(clusters):
            # Extract cluster centroid
            centroid = cluster['centroid']
            
            # Calculate metrics
            distance_to_goal = np.linalg.norm(centroid[:2] - self.goal_position[:2])
            cluster_size = len(cluster['points'])
            
            # Calculate angle between cluster centroid and goal
            vector_to_goal = self.goal_position[:2] - centroid[:2]
            angle_to_goal = np.arctan2(vector_to_goal[1], vector_to_goal[0])
            
            # Compactness of cluster (use average distance from centroid)
            if cluster_size > 1:
                distances = [np.linalg.norm(p[:2] - centroid[:2]) for p in cluster['points']]
                compactness = np.mean(distances)
            else:
                compactness = 0
            
            # Is this a single cow cluster? (originally noise)
            is_single_cow = cluster['label'] == -1 and cluster_size == 1
            
            # Calculate a "blocking" factor - is this cluster blocking other clusters from the goal?
            blocking_factor = 0
            for other_idx, other_cluster in enumerate(clusters):
                if other_idx == idx:
                    continue
                
                other_centroid = other_cluster['centroid']
                other_to_goal = np.linalg.norm(other_centroid[:2] - self.goal_position[:2])
                
                # Is this cluster between the other cluster and the goal?
                if distance_to_goal < other_to_goal:
                    # Vector from other cluster to this one
                    vec_other_to_this = centroid[:2] - other_centroid[:2]
                    # Vector from other cluster to goal
                    vec_other_to_goal = self.goal_position[:2] - other_centroid[:2]
                    
                    # Normalize vectors
                    if np.linalg.norm(vec_other_to_this) > 1e-6 and np.linalg.norm(vec_other_to_goal) > 1e-6:
                        vec_other_to_this = vec_other_to_this / np.linalg.norm(vec_other_to_this)
                        vec_other_to_goal = vec_other_to_goal / np.linalg.norm(vec_other_to_goal)
                        
                        # Dot product gives alignment between vectors
                        alignment = np.dot(vec_other_to_this, vec_other_to_goal)
                        
                        # If alignment is high, this cluster is in the way
                        if alignment > 0.7:  # cos(45°) ≈ 0.7
                            blocking_factor += other_cluster['count']  # Weight by size of blocked cluster
            
            # Calculate priority score
            # Prioritize:
            # 1. Clusters further from goal (higher distance)
            # 2. Larger clusters (more animals)
            # 3. More compact clusters (easier to move)
            # 4. Clusters blocking other clusters' path to goal
            # 5. Single cows get slightly lower priority unless they're far from goal
            priority_score = (
                distance_to_goal * 1.0 +              # Weight for distance to goal
                -cluster_size * 0.5 +                 # Weight for cluster size (negative because larger is better)
                compactness * 0.3 +                   # Weight for compactness
                -blocking_factor * 0.4 +              # Weight for blocking (negative because blocking is important)
                (1.0 if is_single_cow else 0.0) * -2  # Single cows get a penalty unless other factors overwhelm it
            )
            
            # Log the priority calculation
            self.get_logger().debug(
                f"Cluster {idx} priority: {priority_score:.2f} "
                f"(dist={distance_to_goal:.2f}, size={cluster_size}, "
                f"compact={compactness:.2f}, blocking={blocking_factor})"
            )
            
            priorities.append((idx, priority_score))
        
        # Sort by priority score (higher score = higher priority)
        priorities.sort(key=lambda x: -x[1])
        
        return priorities

    def generate_optimal_path(self):
        """
        Generate an optimal path with merging branches where clusters converge toward the goal
        rather than visiting each cluster in sequence.
        
        Returns:
        - Dictionary of path segments with cluster indices as keys
        """
        if not self.prioritized_clusters:
            return {}
        
        # Create a dictionary to store path segments for each cluster
        path_segments = {}
        
        # Track already processed clusters
        processed_clusters = set()
        
        # Goal position
        goal = self.goal_position[:2]
        
        # Process clusters in priority order
        for cluster_idx, _ in self.prioritized_clusters:
            # Skip if already processed
            if cluster_idx in processed_clusters:
                continue
                
            cluster = self.clusters[cluster_idx]
            cluster_centroid = cluster['centroid'][:2]  # Use centroid instead of push position
            
            # Find the best merge point (could be another cluster or the goal)
            best_merge_point = goal
            best_merge_distance = np.linalg.norm(cluster_centroid - goal)
            best_merge_cluster = None
            
            # Check if it's more efficient to merge with another cluster's path
            # rather than going straight to the goal
            for other_idx, _ in self.prioritized_clusters:
                if other_idx == cluster_idx or other_idx in processed_clusters:
                    continue
                    
                other_cluster = self.clusters[other_idx]
                other_centroid = other_cluster['centroid'][:2]
                
                # Vector from current cluster to goal
                vec_to_goal = goal - cluster_centroid
                
                # Vector from current cluster to other cluster
                vec_to_other = other_centroid - cluster_centroid
                
                # Check if other cluster is generally in the direction of the goal
                # by calculating the cosine of the angle between the vectors
                cos_angle = np.dot(vec_to_goal, vec_to_other) / (
                    np.linalg.norm(vec_to_goal) * np.linalg.norm(vec_to_other) + 1e-6)
                
                # If the other cluster is roughly in the direction of the goal (cos_angle > 0.7 means < 45 degrees)
                # and closer than going directly to the goal, merge with it
                if cos_angle > 0.7:
                    distance_through_other = np.linalg.norm(vec_to_other)
                    if distance_through_other < best_merge_distance:
                        best_merge_distance = distance_through_other
                        best_merge_point = other_centroid
                        best_merge_cluster = other_idx
            
            # Create path segment from cluster centroid to best merge point
            if best_merge_cluster is not None:
                # If merging with another cluster, create a path from centroid to that cluster
                path_segments[cluster_idx] = {
                    'points': [cluster_centroid, best_merge_point],
                    'merge_with': best_merge_cluster
                }
            else:
                # If going straight to goal, create a direct path
                path_segments[cluster_idx] = {
                    'points': [cluster_centroid, goal],
                    'merge_with': None
                }
            
            processed_clusters.add(cluster_idx)
        
        return path_segments
    
    def clean_old_markers(self, marker_array):
        """
        Create DELETE markers for all old markers that are no longer needed
        """
        # Add deletion markers for all previous markers
        for ns_name, id_set in self.previous_cluster_markers.items():
            for marker_id in id_set:
                delete_marker = Marker()
                delete_marker.header.frame_id = "map"
                delete_marker.header.stamp = self.get_clock().now().to_msg()
                delete_marker.ns = ns_name
                delete_marker.id = marker_id
                delete_marker.action = Marker.DELETE
                
                marker_array.markers.append(delete_marker)
        
        # Clear all sets for this update cycle
        for ns_name in self.previous_cluster_markers:
            self.previous_cluster_markers[ns_name] = set()
    
    def visualize_target_positions(self, marker_array):
        """
        Create markers to visualize the target positions with big circles
        """
        # Only proceed if we have prioritized clusters and path segments
        if not self.prioritized_clusters or not self.path_segments:
            return
            
        # The current target is the highest priority cluster
        current_target_idx = self.prioritized_clusters[0][0]
        current_target_cluster = self.clusters[current_target_idx]
        current_target_position = current_target_cluster['centroid']
        
        # Create marker for current target
        current_target_marker = Marker()
        current_target_marker.header.frame_id = "map"
        current_target_marker.header.stamp = self.get_clock().now().to_msg()
        current_target_marker.ns = "target_positions"
        current_target_marker.id = 0
        current_target_marker.type = Marker.CYLINDER
        current_target_marker.action = Marker.ADD
        
        # Set position
        current_target_marker.pose.position.x = current_target_position[0]
        current_target_marker.pose.position.y = current_target_position[1]
        current_target_marker.pose.position.z = 0.0
        
        # Set orientation (no rotation)
        current_target_marker.pose.orientation.w = 1.0
        
        # Set scale for a big circle
        current_target_marker.scale.x = 3.0  # Diameter
        current_target_marker.scale.y = 3.0  # Diameter
        current_target_marker.scale.z = 0.1  # Height
        
        # Set color (bright green for current target)
        current_target_marker.color.r = 0.0
        current_target_marker.color.g = 1.0
        current_target_marker.color.b = 0.0
        current_target_marker.color.a = 0.5  # Semi-transparent
        
        # Add to array
        marker_array.markers.append(current_target_marker)
        
        # Track the marker
        self.previous_cluster_markers["target_positions"].add(0)
        
        # Add a label for the current target
        target_label = Marker()
        target_label.header.frame_id = "map"
        target_label.header.stamp = self.get_clock().now().to_msg()
        target_label.ns = "target_positions"
        target_label.id = 3
        target_label.type = Marker.TEXT_VIEW_FACING
        target_label.action = Marker.ADD
        
        target_label.pose.position.x = current_target_position[0]
        target_label.pose.position.y = current_target_position[1]
        target_label.pose.position.z = 1.8  # Higher than regular labels
        
        target_label.text = "CURRENT TARGET"
        
        # Set text size
        target_label.scale.z = 0.7
        
        # Set color (bright green for current target)
        target_label.color.r = 0.0
        target_label.color.g = 1.0
        target_label.color.b = 0.0
        target_label.color.a = 1.0
        
        marker_array.markers.append(target_label)
        self.previous_cluster_markers["target_positions"].add(3)
        
        # Find the next target (the cluster the current target is being pushed towards)
        if current_target_idx in self.path_segments:
            next_target_idx = self.path_segments[current_target_idx]['merge_with']
            
            # If there's a next target cluster, create a marker for it
            if next_target_idx is not None:
                next_target_cluster = self.clusters[next_target_idx]
                next_target_position = next_target_cluster['centroid']
                
                # Create marker for next target
                next_target_marker = Marker()
                next_target_marker.header.frame_id = "map"
                next_target_marker.header.stamp = self.get_clock().now().to_msg()
                next_target_marker.ns = "target_positions"
                next_target_marker.id = 1
                next_target_marker.type = Marker.CYLINDER
                next_target_marker.action = Marker.ADD
                
                # Set position
                next_target_marker.pose.position.x = next_target_position[0]
                next_target_marker.pose.position.y = next_target_position[1]
                next_target_marker.pose.position.z = 0.0
                
                # Set orientation (no rotation)
                next_target_marker.pose.orientation.w = 1.0
                
                # Set scale for a big circle
                next_target_marker.scale.x = 3.0  # Diameter (slightly larger)
                next_target_marker.scale.y = 3.0  # Diameter
                next_target_marker.scale.z = 0.1  # Height
                
                # Set color (purple for next target)
                next_target_marker.color.r = 0.8
                next_target_marker.color.g = 0.0
                next_target_marker.color.b = 0.8
                next_target_marker.color.a = 0.5  # Semi-transparent
                
                # Add to array
                marker_array.markers.append(next_target_marker)
                
                # Track the marker
                self.previous_cluster_markers["target_positions"].add(1)
                
                # Add a label for the next target
                next_label = Marker()
                next_label.header.frame_id = "map"
                next_label.header.stamp = self.get_clock().now().to_msg()
                next_label.ns = "target_positions"
                next_label.id = 4
                next_label.type = Marker.TEXT_VIEW_FACING
                next_label.action = Marker.ADD
                
                next_label.pose.position.x = next_target_position[0]
                next_label.pose.position.y = next_target_position[1]
                next_label.pose.position.z = 1.8  # Higher than regular labels
                
                next_label.text = "NEXT TARGET"
                
                # Set text size
                next_label.scale.z = 0.7
                
                # Set color (purple for next target)
                next_label.color.r = 0.8
                next_label.color.g = 0.0
                next_label.color.b = 0.8
                next_label.color.a = 1.0
                
                marker_array.markers.append(next_label)
                self.previous_cluster_markers["target_positions"].add(4)
            # If the target is being pushed directly to the goal, visualize that
            else:
                # Create a marker for the goal
                goal_marker = Marker()
                goal_marker.header.frame_id = "map"
                goal_marker.header.stamp = self.get_clock().now().to_msg()
                goal_marker.ns = "target_positions"
                goal_marker.id = 2
                goal_marker.type = Marker.CYLINDER
                goal_marker.action = Marker.ADD
                
                # Set position
                goal_marker.pose.position.x = self.goal_position[0]
                goal_marker.pose.position.y = self.goal_position[1]
                goal_marker.pose.position.z = 0.0
                
                # Set orientation (no rotation)
                goal_marker.pose.orientation.w = 1.0
                
                # Set scale for a big circle
                goal_marker.scale.x = 6.0  # Diameter
                goal_marker.scale.y = 6.0  # Diameter
                goal_marker.scale.z = 0.1  # Height
                
                # Set color (blue for goal)
                goal_marker.color.r = 0.0
                goal_marker.color.g = 0.0
                goal_marker.color.b = 1.0
                goal_marker.color.a = 0.5  # Semi-transparent
                
                # Add to array
                marker_array.markers.append(goal_marker)
                
                # Track the marker
                self.previous_cluster_markers["target_positions"].add(2)
                
                # Add a label for the goal
                goal_label = Marker()
                goal_label.header.frame_id = "map"
                goal_label.header.stamp = self.get_clock().now().to_msg()
                goal_label.ns = "target_positions"
                goal_label.id = 5
                goal_label.type = Marker.TEXT_VIEW_FACING
                goal_label.action = Marker.ADD
                
                goal_label.pose.position.x = self.goal_position[0]
                goal_label.pose.position.y = self.goal_position[1]
                goal_label.pose.position.z = 1.8  # Higher than regular labels
                
                goal_label.text = "GOAL"
                
                # Set text size
                goal_label.scale.z = 0.7
                
                # Set color (blue for goal)
                goal_label.color.r = 0.0
                goal_label.color.g = 0.0
                goal_label.color.b = 1.0
                goal_label.color.a = 1.0
                
                marker_array.markers.append(goal_label)
                self.previous_cluster_markers["target_positions"].add(5)
    
    def visualize_optimal_path(self, marker_array):
        """
        Create markers for visualizing the optimal path with merging branches
        """
        if not hasattr(self, 'path_segments') or not self.path_segments:
            # If path_segments doesn't exist yet, compute it from optimal_path
            if hasattr(self, 'optimal_path') and self.optimal_path:
                self.path_segments = self.generate_optimal_path()
            else:
                return
        
        # Create unique marker for each path segment
        for i, (cluster_idx, segment) in enumerate(self.path_segments.items()):
            # Create marker for path segment
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "optimal_path"
            path_marker.id = i
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            
            # Use cluster centroid instead of push position for the starting point
            cluster_centroid = self.clusters[cluster_idx]['centroid'][:2]
            
            # Create points list starting with the cluster centroid
            points = []
            points.append(Point(x=cluster_centroid[0], y=cluster_centroid[1], z=self.wrangler_position[2]))
            
            # Add the second point (target point)
            if len(segment['points']) >= 2:
                target_point = segment['points'][1]  # This is still the destination point
                points.append(Point(x=target_point[0], y=target_point[1], z=self.wrangler_position[2]))
            
            path_marker.points = points
            
            # Set line width
            path_marker.scale.x = 0.15
            
            # Color based on priority - gradient from yellow (highest) to green (lowest)
            priority_pos = [p[0] for p in self.prioritized_clusters].index(cluster_idx)
            priority_ratio = priority_pos / max(1, len(self.prioritized_clusters) - 1)
            
            path_marker.color.r = 0.7 * (1 - priority_ratio)
            path_marker.color.g = 0.9
            path_marker.color.b = 0.2 * priority_ratio
            path_marker.color.a = 0.5
            
            # Add to array and track
            marker_array.markers.append(path_marker)
            self.previous_cluster_markers["optimal_path"].add(i)
        
        # Add final segment to goal if needed
        # (this ensures there's always a visible path to goal)
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "optimal_path"
        goal_marker.id = len(self.path_segments)
        goal_marker.type = Marker.LINE_STRIP
        goal_marker.action = Marker.ADD
        
        # Check if there are any clusters that go directly to goal
        direct_to_goal = False
        for segment in self.path_segments.values():
            if segment['merge_with'] is None:
                direct_to_goal = True
                break
        
        if not direct_to_goal and self.prioritized_clusters:
            # If no cluster goes directly to goal, add highest priority cluster to goal
            highest_priority_idx = self.prioritized_clusters[0][0]
            highest_priority_centroid = self.clusters[highest_priority_idx]['centroid']
            
            goal_marker.points = [
                Point(x=highest_priority_centroid[0], y=highest_priority_centroid[1], z=self.wrangler_position[2]),
                Point(x=self.goal_position[0], y=self.goal_position[1], z=self.goal_position[2])
            ]
            
            goal_marker.scale.x = 0.15
            goal_marker.color.r = 0.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 1.0
            
            marker_array.markers.append(goal_marker)
            self.previous_cluster_markers["optimal_path"].add(len(self.path_segments))
        
        # Add current position to closest path segment
        wrangler_marker = Marker()
        wrangler_marker.header.frame_id = "map"
        wrangler_marker.header.stamp = self.get_clock().now().to_msg()
        wrangler_marker.ns = "optimal_path"
        wrangler_marker.id = len(self.path_segments) + 1
        wrangler_marker.type = Marker.LINE_STRIP
        wrangler_marker.action = Marker.ADD
        
        # Find closest cluster centroid to connect wrangler position
        closest_point = None
        closest_distance = float('inf')
        
        for cluster in self.clusters:
            centroid = cluster['centroid'][:2]
            dist = np.linalg.norm(self.wrangler_position[:2] - centroid)
            if dist < closest_distance:
                closest_distance = dist
                closest_point = centroid
        
        if closest_point is not None:
            wrangler_marker.points = [
                Point(x=self.wrangler_position[0], y=self.wrangler_position[1], z=self.wrangler_position[2]),
                Point(x=closest_point[0], y=closest_point[1], z=self.wrangler_position[2])
            ]
            
            wrangler_marker.scale.x = 0.1
            wrangler_marker.color.r = 1.0
            wrangler_marker.color.g = 0.5
            wrangler_marker.color.b = 0.0
            wrangler_marker.color.a = 0.6
            
            marker_array.markers.append(wrangler_marker)
            self.previous_cluster_markers["optimal_path"].add(len(self.path_segments) + 1)
        
        # Add sequence numbers to each cluster centroid
        for i, (cluster_idx, _) in enumerate(self.prioritized_clusters):
            cluster_centroid = self.clusters[cluster_idx]['centroid']
            
            # Create text marker for sequence number
            sequence_text = Marker()
            sequence_text.header.frame_id = "map"
            sequence_text.header.stamp = self.get_clock().now().to_msg()
            sequence_text.ns = "optimal_path"
            sequence_text.id = len(self.path_segments) + 10 + i  # Use higher IDs for text
            sequence_text.type = Marker.TEXT_VIEW_FACING
            sequence_text.action = Marker.ADD
            
            # Position text at cluster centroid
            sequence_text.pose.position.x = cluster_centroid[0]
            sequence_text.pose.position.y = cluster_centroid[1]
            sequence_text.pose.position.z = self.wrangler_position[2] + 0.5  # Slightly above path
            
            sequence_text.text = f"{i+1}"  # Sequence number
            
            # Set text size
            sequence_text.scale.z = 0.6
            
            # Set color (white text with black outline)
            sequence_text.color.r = 1.0
            sequence_text.color.g = 1.0
            sequence_text.color.b = 1.0
            sequence_text.color.a = 1.0
            
            # Add to array and track
            marker_array.markers.append(sequence_text)
            self.previous_cluster_markers["optimal_path"].add(len(self.path_segments) + 10 + i)

    def publish_optimal_path(self):
        """
        Publish the planned optimal mustering path with merging branches
        """
        # Calculate path segments if not already done
        if not hasattr(self, 'path_segments') or not self.path_segments:
            self.path_segments = self.generate_optimal_path()
            
        if not self.path_segments:
            # If no path segments yet, create a simple path to goal
            optimal_path = Path()
            optimal_path.header.frame_id = "map"
            optimal_path.header.stamp = self.get_clock().now().to_msg()
            
            # Start from current wrangler position
            timestamp = self.get_clock().now().to_msg()
            current_pose = PoseStamped()
            current_pose.header.frame_id = "map"
            current_pose.header.stamp = timestamp
            current_pose.pose.position.x = self.wrangler_position[0]
            current_pose.pose.position.y = self.wrangler_position[1]
            current_pose.pose.position.z = self.wrangler_position[2]
            current_pose.pose.orientation.w = 1.0
            
            # Add goal position
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = timestamp
            goal_pose.pose.position.x = self.goal_position[0]
            goal_pose.pose.position.y = self.goal_position[1]
            goal_pose.pose.position.z = self.goal_position[2]
            goal_pose.pose.orientation.w = 1.0
            
            optimal_path.poses = [current_pose, goal_pose]
            
            # Publish simple path
            self.optimal_path_pub.publish(optimal_path)
            return
            
        optimal_path = Path()
        optimal_path.header.frame_id = "map"
        optimal_path.header.stamp = self.get_clock().now().to_msg()
        
        # Start from current position
        timestamp = self.get_clock().now().to_msg()
        current_pose = PoseStamped()
        current_pose.header.frame_id = "map"
        current_pose.header.stamp = timestamp
        current_pose.pose.position.x = self.wrangler_position[0]
        current_pose.pose.position.y = self.wrangler_position[1]
        current_pose.pose.position.z = self.wrangler_position[2]
        current_pose.pose.orientation.w = 1.0
        
        optimal_path.poses.append(current_pose)
        
        # Process clusters in priority order and add their path segments
        # This ensures the highest priority branches appear first in the path
        processed_clusters = set()
        
        for cluster_idx, _ in self.prioritized_clusters:
            if cluster_idx in processed_clusters:
                continue
                
            # Start with the cluster centroid, not the push position
            cluster_centroid = self.clusters[cluster_idx]['centroid']
            
            # Add the cluster centroid to the path
            centroid_pose = PoseStamped()
            centroid_pose.header.frame_id = "map"
            centroid_pose.header.stamp = timestamp
            centroid_pose.pose.position.x = cluster_centroid[0]
            centroid_pose.pose.position.y = cluster_centroid[1]
            centroid_pose.pose.position.z = self.wrangler_position[2]
            centroid_pose.pose.orientation.w = 1.0
            
            optimal_path.poses.append(centroid_pose)
            
            # Follow the path from this cluster to its eventual goal or merge
            current_idx = cluster_idx
            while current_idx is not None and current_idx not in processed_clusters:
                # Mark as processed
                processed_clusters.add(current_idx)
                
                # If this cluster merges with another, add the target cluster's centroid
                if current_idx in self.path_segments:
                    segment = self.path_segments[current_idx]
                    
                    # If this segment merges with another cluster, add that cluster's centroid
                    if segment['merge_with'] is not None:
                        next_idx = segment['merge_with']
                        next_centroid = self.clusters[next_idx]['centroid']
                        
                        next_pose = PoseStamped()
                        next_pose.header.frame_id = "map"
                        next_pose.header.stamp = timestamp
                        next_pose.pose.position.x = next_centroid[0]
                        next_pose.pose.position.y = next_centroid[1]
                        next_pose.pose.position.z = self.wrangler_position[2]
                        next_pose.pose.orientation.w = 1.0
                        
                        optimal_path.poses.append(next_pose)
                    else:
                        # This segment goes directly to goal, so add the goal point
                        goal_pose = PoseStamped()
                        goal_pose.header.frame_id = "map"
                        goal_pose.header.stamp = timestamp
                        goal_pose.pose.position.x = self.goal_position[0]
                        goal_pose.pose.position.y = self.goal_position[1]
                        goal_pose.pose.position.z = self.goal_position[2]
                        goal_pose.pose.orientation.w = 1.0
                        
                        optimal_path.poses.append(goal_pose)
                    
                    # Move to next segment if merging
                    current_idx = segment['merge_with']
                else:
                    current_idx = None
        
        # Add the goal position if not already added
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = timestamp
        goal_pose.pose.position.x = self.goal_position[0]
        goal_pose.pose.position.y = self.goal_position[1]
        goal_pose.pose.position.z = self.goal_position[2]
        goal_pose.pose.orientation.w = 1.0
        
        # Ensure goal isn't duplicated if already in the path
        last_pose = optimal_path.poses[-1] if optimal_path.poses else None
        if (not last_pose or 
            abs(last_pose.pose.position.x - self.goal_position[0]) > 0.1 or
            abs(last_pose.pose.position.y - self.goal_position[1]) > 0.1):
            optimal_path.poses.append(goal_pose)
        
        # Publish the optimal path
        self.optimal_path_pub.publish(optimal_path)
    
    def update_visualization(self):
        """
        Update visualization markers for cattle, clusters, and optimal mustering path
        """
        # Always attempt visualization, no early return now
        
        # Create marker array message
        marker_array = MarkerArray()
        
        # First, clean up any old markers
        self.clean_old_markers(marker_array)
        
        # Add cattle markers
        for i, (cow_id, position) in enumerate(self.cattle_positions.items()):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cattle"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            
            # Set orientation (no rotation)
            marker.pose.orientation.w = 1.0
            
            # Set scale
            marker.scale.x = 1.0  # Cattle width
            marker.scale.y = 1.0  # Cattle length
            marker.scale.z = 1.0  # Cattle height
            
            # Set color (brown for cattle)
            marker.color.r = 0.6
            marker.color.g = 0.4
            marker.color.b = 0.2
            marker.color.a = 1.0
            
            # Add to array
            marker_array.markers.append(marker)
        
        # Add wrangler/drone marker
        wrangler_marker = Marker()
        wrangler_marker.header.frame_id = "map"
        wrangler_marker.header.stamp = self.get_clock().now().to_msg()
        wrangler_marker.ns = "wrangler"
        wrangler_marker.id = 0
        wrangler_marker.type = Marker.CUBE
        wrangler_marker.action = Marker.ADD
        
        # Set position
        wrangler_marker.pose.position.x = self.wrangler_position[0]
        wrangler_marker.pose.position.y = self.wrangler_position[1]
        wrangler_marker.pose.position.z = self.wrangler_position[2]
        
        # Set orientation (no rotation)
        wrangler_marker.pose.orientation.w = 1.0
        
        # Set scale
        wrangler_marker.scale.x = 0.5
        wrangler_marker.scale.y = 0.5
        wrangler_marker.scale.z = 0.2
        
        # Set color (red for wrangler/drone)
        wrangler_marker.color.r = 1.0
        wrangler_marker.color.g = 0.0
        wrangler_marker.color.b = 0.0
        wrangler_marker.color.a = 1.0
        
        # Add to array
        marker_array.markers.append(wrangler_marker)
        
        # Add height visualization (vertical line from ground to wrangler)
        height_marker = Marker()
        height_marker.header.frame_id = "map"
        height_marker.header.stamp = self.get_clock().now().to_msg()
        height_marker.ns = "wrangler_height"
        height_marker.id = 0
        height_marker.type = Marker.LINE_STRIP
        height_marker.action = Marker.ADD
        
        # Create vertical line
        ground_point = Point(x=self.wrangler_position[0], y=self.wrangler_position[1], z=0.0)
        wrangler_point = Point(x=self.wrangler_position[0], y=self.wrangler_position[1], z=self.wrangler_position[2])
        
        height_marker.points = [ground_point, wrangler_point]
        
        # Set line width
        height_marker.scale.x = 0.05
        
        # Set color (yellow for height line)
        height_marker.color.r = 1.0
        height_marker.color.g = 1.0
        height_marker.color.b = 0.0
        height_marker.color.a = 0.8
        
        # Add to array
        marker_array.markers.append(height_marker)
        
        # Add height text marker
        height_text = Marker()
        height_text.header.frame_id = "map"
        height_text.header.stamp = self.get_clock().now().to_msg()
        height_text.ns = "wrangler_height_text"
        height_text.id = 0
        height_text.type = Marker.TEXT_VIEW_FACING
        height_text.action = Marker.ADD
        
        # Position text next to the height line
        height_text.pose.position.x = self.wrangler_position[0] + 0.5
        height_text.pose.position.y = self.wrangler_position[1]
        height_text.pose.position.z = max(self.wrangler_position[2] / 2.0, 0.5)
        
        height_text.text = f"Height: {self.wrangler_position[2]:.2f}m"
        
        # Set text size
        height_text.scale.z = 0.4
        
        # Set color (white text)
        height_text.color.r = 1.0
        height_text.color.g = 1.0
        height_text.color.b = 1.0
        height_text.color.a = 1.0
        
        # Add to array
        marker_array.markers.append(height_text)
        
        # Reset current cluster count for this cycle
        self.current_cluster_count = 0
        self.clusters = []  # Reset clusters data
        
        # Perform DBSCAN clustering
        if len(self.cattle_positions) >= self.min_samples:
            # Extract cattle positions and IDs
            cow_ids = list(self.cattle_positions.keys())
            positions = np.array([self.cattle_positions[cow_id][:2] for cow_id in cow_ids])
            
            # Perform clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(positions)
            labels = clustering.labels_
            
            # Clear old cluster data and create new mapping
            self.cluster_data = {}
            
            # Create convex hulls and analyze each cluster
            unique_labels = set(labels)
            cluster_idx = 0  # Use continuous numbering for cluster visualization
            
            # First handle all regular clusters
            for label in unique_labels:
                if label == -1:
                    # We'll handle noise points separately - don't skip them completely
                    continue
                
                # Get points in this cluster
                cluster_mask = labels == label
                cluster_points = positions[cluster_mask]
                cluster_cow_ids = [cow_ids[i] for i, in_cluster in enumerate(cluster_mask) if in_cluster]
                
                # Get full position data (including z coordinate)
                cluster_positions = [self.cattle_positions[cow_id] for cow_id in cluster_cow_ids]
                
                # Calculate centroid
                centroid = np.mean(cluster_positions, axis=0)
                
                # Store cluster information
                self.clusters.append({
                    'label': label,
                    'index': cluster_idx,
                    'cow_ids': cluster_cow_ids,
                    'points': cluster_positions,
                    'centroid': centroid,
                    'count': len(cluster_points)
                })
                
                # Log cluster information for debugging
                self.get_logger().debug(f"Cluster {cluster_idx}: {len(cluster_points)} cattle, IDs: {cluster_cow_ids}")
                
                # Skip if not enough points for convex hull (need at least 3)
                if len(cluster_points) < 3:
                    # Just draw a line or point if we can't make a convex hull
                    if len(cluster_points) == 2:
                        # Draw a line between the two points
                        line_marker = Marker()
                        line_marker.header.frame_id = "map"
                        line_marker.header.stamp = self.get_clock().now().to_msg()
                        line_marker.ns = "cluster_line"
                        line_marker.id = cluster_idx
                        line_marker.type = Marker.LINE_STRIP
                        line_marker.action = Marker.ADD
                        
                        # Add the points
                        p1 = Point(x=cluster_points[0, 0], y=cluster_points[0, 1], z=0.0)
                        p2 = Point(x=cluster_points[1, 0], y=cluster_points[1, 1], z=0.0)
                        line_marker.points = [p1, p2]
                        
                        # Set line width
                        line_marker.scale.x = 0.1
                        
                        # Set color
                        line_marker.color.r = 0.0
                        line_marker.color.g = 0.5
                        line_marker.color.b = 1.0
                        line_marker.color.a = 1.0
                        
                        # Add to array and track
                        marker_array.markers.append(line_marker)
                        self.previous_cluster_markers["cluster_line"].add(cluster_idx)
                    
                    # Always add text for any non-noise cluster
                    text_marker = Marker()
                    text_marker.header.frame_id = "map"
                    text_marker.header.stamp = self.get_clock().now().to_msg()
                    text_marker.ns = "cluster_text"
                    text_marker.id = cluster_idx
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    
                    # Position text at the center of the cluster
                    center = np.mean(cluster_points, axis=0)
                    text_marker.pose.position.x = center[0]
                    text_marker.pose.position.y = center[1]
                    text_marker.pose.position.z = 1.5  # Put above the cattle
                    
                    text_marker.text = f"Cluster {cluster_idx}: {len(cluster_points)} cattle"
                    
                    # Set text size
                    text_marker.scale.z = 0.5
                    
                    # Set color (white text)
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0
                    
                    # Add to array and track
                    marker_array.markers.append(text_marker)
                    self.previous_cluster_markers["cluster_text"].add(cluster_idx)
                    
                    # Increment cluster index for consistent visualization
                    cluster_idx += 1
                    continue
                
                # Create convex hull for the cluster
                try:
                    hull = ConvexHull(cluster_points)
                    
                    # Create hull marker
                    hull_marker = Marker()
                    hull_marker.header.frame_id = "map"
                    hull_marker.header.stamp = self.get_clock().now().to_msg()
                    hull_marker.ns = "cluster_hull"
                    hull_marker.id = cluster_idx
                    hull_marker.type = Marker.LINE_STRIP
                    hull_marker.action = Marker.ADD
                    
                    # Create line strip for convex hull
                    # Follow the vertices in the order computed by ConvexHull
                    hull_points = []
                    for vertex_idx in hull.vertices:
                        p = Point(x=cluster_points[vertex_idx, 0], 
                                 y=cluster_points[vertex_idx, 1], 
                                 z=0.0)
                        hull_points.append(p)
                    
                    # Close the loop
                    hull_points.append(hull_points[0])
                    hull_marker.points = hull_points
                    
                    # Set line width
                    hull_marker.scale.x = 0.1
                    
                    # Set color (blue for cluster hull)
                    hull_marker.color.r = 0.0
                    hull_marker.color.g = 0.5
                    hull_marker.color.b = 1.0
                    hull_marker.color.a = 1.0
                    
                    # Add to array and track
                    marker_array.markers.append(hull_marker)
                    self.previous_cluster_markers["cluster_hull"].add(cluster_idx)
                    
                    # Create a filled polygon for the hull (semi-transparent)
                    hull_fill = Marker()
                    hull_fill.header.frame_id = "map"
                    hull_fill.header.stamp = self.get_clock().now().to_msg()
                    hull_fill.ns = "cluster_hull_fill"
                    hull_fill.id = cluster_idx
                    hull_fill.type = Marker.TRIANGLE_LIST
                    hull_fill.action = Marker.ADD
                    
                    # Create triangles for the convex hull (fan triangulation from centroid)
                    centroid = np.mean(cluster_points, axis=0)
                    centroid_point = Point(x=centroid[0], y=centroid[1], z=0.0)
                    
                    # For each edge in the hull, create a triangle with the centroid
                    for i in range(len(hull.vertices)):
                        idx1 = hull.vertices[i]
                        idx2 = hull.vertices[(i + 1) % len(hull.vertices)]
                        
                        p1 = Point(x=cluster_points[idx1, 0], y=cluster_points[idx1, 1], z=0.0)
                        p2 = Point(x=cluster_points[idx2, 0], y=cluster_points[idx2, 1], z=0.0)
                        
                        # Add centroid and two hull points to create a triangle
                        hull_fill.points.extend([centroid_point, p1, p2])
                    
                    # Set color (semi-transparent blue fill)
                    hull_fill.color.r = 0.0
                    hull_fill.color.g = 0.3
                    hull_fill.color.b = 0.8
                    hull_fill.color.a = 0.3  # Semi-transparent
                    
                    # Set scale (for triangle thickness)
                    hull_fill.scale.x = 1.0
                    hull_fill.scale.y = 1.0
                    hull_fill.scale.z = 1.0
                    
                    # Add to array and track
                    marker_array.markers.append(hull_fill)
                    self.previous_cluster_markers["cluster_hull_fill"].add(cluster_idx)
                    
                    # Add cluster label (number of cattle in cluster)
                    text_marker = Marker()
                    text_marker.header.frame_id = "map"
                    text_marker.header.stamp = self.get_clock().now().to_msg()
                    text_marker.ns = "cluster_text"
                    text_marker.id = cluster_idx
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    
                    # Position text above the center of the cluster
                    text_marker.pose.position.x = centroid[0]
                    text_marker.pose.position.y = centroid[1]
                    text_marker.pose.position.z = 1.5  # Put above the cattle
                    
                    text_marker.text = f"Cluster {cluster_idx}: {len(cluster_points)} cattle"
                    
                    # Set text size
                    text_marker.scale.z = 0.5
                    
                    # Set color (white text)
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0
                    
                    # Add to array and track
                    marker_array.markers.append(text_marker)
                    self.previous_cluster_markers["cluster_text"].add(cluster_idx)
                    
                except Exception as e:
                    self.get_logger().warn(f"Failed to create convex hull for cluster {cluster_idx}: {e}")
                    # Fall back to a simpler visualization if convex hull fails
                
                # Increment cluster index for consistent visualization
                cluster_idx += 1
            
            # Now handle any noise points (individual cows not in clusters)
            # These are labeled as -1 in DBSCAN
            noise_mask = labels == -1
            noise_indices = np.where(noise_mask)[0]
            
            for i in noise_indices:
                # Get the individual cow position
                cow_id = cow_ids[i]
                position = positions[i]
                
                # Create a "cluster" for this single cow
                self.clusters.append({
                    'label': -1,  # Mark as originally noise
                    'index': cluster_idx,
                    'cow_ids': [cow_id],
                    'points': [self.cattle_positions[cow_id]],
                    'centroid': self.cattle_positions[cow_id],
                    'count': 1
                })
                
                # Log individual cow "cluster"
                self.get_logger().debug(f"Individual cow cluster {cluster_idx}: ID: {cow_id}")
                
                # Create marker for individual cow "cluster"
                text_marker = Marker()
                text_marker.header.frame_id = "map"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = "cluster_text"
                text_marker.id = cluster_idx
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                # Position text above the cow
                text_marker.pose.position.x = position[0]
                text_marker.pose.position.y = position[1]
                text_marker.pose.position.z = 1.5  # Put above the cattle
                
                text_marker.text = f"Cow {cluster_idx}: {cow_id}"
                
                # Set text size
                text_marker.scale.z = 0.4
                
                # Set color (white text)
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                # Add to array and track
                marker_array.markers.append(text_marker)
                self.previous_cluster_markers["cluster_text"].add(cluster_idx)
                
                # Increment cluster index
                cluster_idx += 1
                
            # Update the current cluster count
            self.current_cluster_count = cluster_idx
            
            # Prioritize clusters for mustering
            if self.clusters:
                # Calculate new cluster priorities
                new_prioritized_clusters = self.calculate_cluster_metrics(self.clusters)
                
                # Check if we need to update the path planning
                update_path_planning = self.check_significant_weight_changes(new_prioritized_clusters)
                
                if update_path_planning:
                    # Store the new prioritization
                    self.prioritized_clusters = new_prioritized_clusters
                
                    # Generate optimal path through push positions with merging branches
                    self.path_segments = self.generate_optimal_path()
                    
                    # Since we're not updating the path planning, need to refresh the push positions and directions
                    # for the new clusters but keep the same path structure
                    self.get_logger().debug("Skipping path planning update, only refreshing positions")

                # Visualize optimal path
                self.visualize_optimal_path(marker_array)
                
                # Visualize target positions with big circles
                self.visualize_target_positions(marker_array)
        
        # Publish marker array
        self.marker_pub.publish(marker_array)
        
        # Update goal visualization
        self.publish_goal()
        
        # Update path visualization
        self.publish_path()
        
        # Update optimal path visualization
        self.publish_optimal_path()
        
        # Publish target positions
        self.publish_target_positions()

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedCattleVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()