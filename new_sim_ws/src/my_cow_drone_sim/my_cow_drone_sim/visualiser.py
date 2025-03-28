#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import math
from std_msgs.msg import ColorRGBA

class CattleVisualizerNode(Node):
    """
    ROS2 node for visualizing cattle in RViz with cluster convex hulls,
    the path to the goal, and the height of the wrangler.
    """
    
    def __init__(self):
        super().__init__('cattle_visualizer_node')
        
        # Parameters
        self.num_cows = 12
        self.update_rate = 10  # Hz
        
        # DBSCAN parameters
        self.eps = 2.0        # Maximum distance between two samples to be in same cluster
        self.min_samples = 2  # Minimum number of samples in a cluster
        
        # State tracking
        self.cattle_positions = {}
        self.wrangler_position = np.array([0.0, 0.0, 0.0])
        self.previous_wrangler_positions = []  # To track path
        self.max_path_length = 100  # Maximum number of points to keep in the path
        
        # Track active cluster markers for cleanup
        self.current_cluster_count = 0
        self.previous_cluster_markers = {
            "cluster_hull": set(),
            "cluster_hull_fill": set(),
            "cluster_text": set(),
            "cluster_line": set()
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
        
        # Subscribe to cattle odometry
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
        
        # Subscribe to wrangler/drone odometry
        self.create_subscription(
            Odometry,
            '/drone/odom',  # Assuming this is the drone topic
            self.wrangler_callback,
            10
        )
        
        # Timer for visualization updates
        self.update_timer = self.create_timer(1.0/self.update_rate, self.update_visualization)
        
        # Set goal position (same as in the wrangler node)
        self.goal_position = np.array([7.0, 7.0, 0.0])
        
        # Publish initial goal visualization
        self.publish_goal()
        
        # Store cluster data for debugging
        self.cluster_data = {}
        
        self.get_logger().info('Enhanced Cattle Visualizer node initialized with convex hull')
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions from odometry
        """
        pos = msg.pose.pose.position
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
    
    def wrangler_callback(self, msg):
        """
        Update wrangler position from odometry and store for path visualization
        """
        pos = msg.pose.pose.position
        self.wrangler_position = np.array([pos.x, pos.y, pos.z])
        
        # Log height for debugging
        self.get_logger().debug(f'Wrangler height: {pos.z}')
        
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
    
    def publish_path(self):
        """
        Publish the path from wrangler's current position to the goal
        """
        # First publish the historical path of the wrangler
        historical_path = Path()
        historical_path.header.frame_id = "map"
        historical_path.header.stamp = self.get_clock().now().to_msg()
        historical_path.poses = self.previous_wrangler_positions
        
        self.path_pub.publish(historical_path)
        
        # Now create a marker for the line to the goal
        marker_array = MarkerArray()
        
        path_to_goal = Marker()
        path_to_goal.header.frame_id = "map"
        path_to_goal.header.stamp = self.get_clock().now().to_msg()
        path_to_goal.ns = "path_to_goal"
        path_to_goal.id = 0
        path_to_goal.type = Marker.LINE_STRIP
        path_to_goal.action = Marker.ADD
        
        # Create line from current position to goal
        start_point = Point(x=self.wrangler_position[0], y=self.wrangler_position[1], z=self.wrangler_position[2])
        end_point = Point(x=self.goal_position[0], y=self.goal_position[1], z=self.goal_position[2])
        
        path_to_goal.points = [start_point, end_point]
        
        # Set line width
        path_to_goal.scale.x = 0.1
        
        # Set color (green for path to goal)
        path_to_goal.color.r = 0.0
        path_to_goal.color.g = 1.0
        path_to_goal.color.b = 0.0
        path_to_goal.color.a = 0.8
        
        # Add to array
        marker_array.markers.append(path_to_goal)
        
        # Publish the direct path marker
        self.marker_pub.publish(marker_array)
    
    def find_furthest_cow(self):
        """
        Find the cow that is furthest from the goal
        """
        max_distance = -1
        furthest_cow_id = None
        
        for cow_id, position in self.cattle_positions.items():
            distance = np.linalg.norm(position[:2] - self.goal_position[:2])
            if distance > max_distance:
                max_distance = distance
                furthest_cow_id = cow_id
                
        return furthest_cow_id
    
    def clean_old_markers(self, marker_array):
        """
        Create DELETE markers for all old cluster markers that are no longer needed
        """
        # Add deletion markers for all previous cluster markers
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
        
    def update_visualization(self):
        """
        Update visualization markers for cattle, cluster convex hulls,
        and wrangler height
        """
        # Skip if we don't have enough data yet
        if len(self.cattle_positions) < 1:
            return
        
        # Find the furthest cow
        furthest_cow_id = self.find_furthest_cow()
        
        # Create marker array message
        marker_array = MarkerArray()
        
        # First, clean up any old cluster markers
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
            marker.scale.x = 0.7  # Cattle width
            marker.scale.y = 1.2  # Cattle length
            marker.scale.z = 1.0  # Cattle height
            
            # Set color (purple for furthest cow, brown for others)
            if cow_id == furthest_cow_id:
                marker.color.r = 0.6
                marker.color.g = 0.0
                marker.color.b = 0.8  # Purple for furthest cow
                marker.color.a = 1.0
            else:
                marker.color.r = 0.6
                marker.color.g = 0.4
                marker.color.b = 0.2  # Brown for regular cattle
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
        height_text.pose.position.z = max(self.wrangler_position[2] / 2.0, 0.5)  # Midpoint of height, at least 0.5m
        
        # Make sure we're displaying the actual z value from the odometry message
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
            
            # Create convex hulls for each cluster
            unique_labels = set(labels)
            cluster_idx = 0  # Use continuous numbering for cluster visualization
            
            for label in unique_labels:
                if label == -1:
                    # Skip noise points
                    continue
                
                # Get points in this cluster
                cluster_mask = labels == label
                cluster_points = positions[cluster_mask]
                cluster_cow_ids = [cow_ids[i] for i, in_cluster in enumerate(cluster_mask) if in_cluster]
                
                # Store the cluster information for debugging
                self.cluster_data[label] = {
                    'cow_ids': cluster_cow_ids,
                    'positions': cluster_points.tolist(),
                    'count': len(cluster_points)
                }
                
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
                
            # Update the current cluster count
            self.current_cluster_count = cluster_idx
        
        # Publish marker array
        self.marker_pub.publish(marker_array)
        
        # Update goal visualization
        self.publish_goal()
        
        # Update path visualization
        self.publish_path()

def main(args=None):
    rclpy.init(args=args)
    node = CattleVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()