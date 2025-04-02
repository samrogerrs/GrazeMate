#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
import math


class FixedDirectionWranglerNode(Node):
    """
    Corrected ROS2 node for drone control that properly targets the far edge of clusters
    regardless of clockwise/counterclockwise movement.
    """
    
    def __init__(self):
        super().__init__('wrangler_node')
        
        # Basic configuration parameters
        self.operational_height = 0.5  # Normal operational height (m)
        self.transit_height = 4.0      # Higher height for transit to avoid scaring cattle (m)
        self.speed = 4.0               # Base movement speed (m/s)
        self.min_distance = 0.5        # Minimum distance to maintain from target (m)
        
        # Edge targeting parameters
        self.edge_offset = 1.0         # Additional distance beyond cluster radius to target (m)
        
        # Position threshold - when to consider the drone "in position"
        self.position_threshold = 3.0  # Distance to target when considered "in position" (m)
        self.descent_rate = 2.0        # How fast to descend when in position (m/s)
        
        # Publishers
        self.velocity_pub = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/wrangler/status', 10)
        
        # State tracking
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.drone_orientation = 0.0
        self.is_in_position = False    # Flag to track if drone is in position
        
        # Target tracking
        self.current_target_position = None
        self.next_target_position = None
        self.target_cluster_radius = 2.5  # Default radius
        self.calculated_edge_target = None # The calculated far edge target position
        
        # Setup core subscribers
        self.setup_subscribers()
        
        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Direction-fixed Wrangler node initialized')
    
    def setup_subscribers(self):
        """Set up essential subscribers"""
        # Subscribe to drone odometry with multiple topic options
        for topic in ['/drone/odom', '/wrangler/odom', '/robot/odom']:
            self.create_subscription(
                Odometry,
                topic,
                self.drone_callback,
                10
            )
        
        # Subscribe to marker array for target cylinders
        self.create_subscription(
            MarkerArray,
            '/visualization/markers',
            self.markers_callback,
            10
        )
        
        # Direct subscriptions to target points
        self.create_subscription(
            PointStamped,
            '/visualization/current_target',
            self.current_target_callback,
            10
        )
        
        self.create_subscription(
            PointStamped,
            '/visualization/next_target',
            self.next_target_callback,
            10
        )
    
    def drone_callback(self, msg):
        """Update drone position and orientation from odometry"""
        # Update position
        pos = msg.pose.pose.position
        self.drone_position = np.array([pos.x, pos.y, pos.z])
        
        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        self.drone_orientation = math.atan2(siny_cosp, cosy_cosp)
    
    def current_target_callback(self, msg):
        """Process the current target point directly from topic"""
        self.current_target_position = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().debug(f"Received current target at {self.current_target_position}")
        self.calculate_far_edge_target()
    
    def next_target_callback(self, msg):
        """Process the next target point directly from topic"""
        self.next_target_position = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().debug(f"Received next target at {self.next_target_position}")
        self.calculate_far_edge_target()
    
    def markers_callback(self, msg):
        """
        Extract target positions from cylinder markers
        """
        for marker in msg.markers:
            # Only process cylinder markers from the target_positions namespace
            if marker.ns == "target_positions" and marker.type == Marker.CYLINDER:
                # Current target is ID 0
                if marker.id == 0:
                    self.current_target_position = np.array([
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ])
                    # Extract radius from marker scale
                    self.target_cluster_radius = marker.scale.x / 2.0
                    self.get_logger().debug(f"Found target at {self.current_target_position}, radius: {self.target_cluster_radius}")
                
                # Next target is ID 1 or goal is ID 2
                elif marker.id in [1, 2]:
                    self.next_target_position = np.array([
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ])
        
        # After processing markers, recalculate the far edge target
        self.calculate_far_edge_target()
    
    def calculate_far_edge_target(self):
        """
        Calculate a target position on the far edge of the current cluster,
        opposite from the direction of the next target.
        """
        if self.current_target_position is None:
            self.calculated_edge_target = None
            return
            
        # If we don't have a next target, use a default direction (positive X axis)
        if self.next_target_position is None:
            # Just move along positive X axis by default
            herd_direction = np.array([1.0, 0.0])
            self.get_logger().debug("No next target, using default direction for edge targeting")
        else:
            # Calculate the vector FROM current target TO next target
            # This is the direction we want to herd the cattle
            herd_direction = self.next_target_position[:2] - self.current_target_position[:2]
            norm = np.linalg.norm(herd_direction)
            
            if norm > 0.001:
                herd_direction = herd_direction / norm
            else:
                herd_direction = np.array([1.0, 0.0])  # Default if targets are too close
                
            self.get_logger().debug(f"Herd direction: {herd_direction}")
        
        # The approach direction is OPPOSITE to the herd direction
        # We want to be on the far side of the cluster from the next target
        approach_direction = -herd_direction
        
        # Calculate point on far edge of cluster (center + approach_direction * (radius + offset))
        edge_target = self.current_target_position[:2] + approach_direction * (self.target_cluster_radius + self.edge_offset)
        
        # Create complete target position with original Z coordinate
        self.calculated_edge_target = np.array([
            edge_target[0],
            edge_target[1],
            self.current_target_position[2]
        ])
        
        self.get_logger().debug(f"Calculated far edge target at {self.calculated_edge_target}")
    
    def publish_status(self, message):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def control_loop(self):
        """
        Control loop: target the far edge of the cluster
        """
        # Create velocity message
        cmd_vel = Twist()
        
        # If we have no target or no calculated edge target, hover in place
        if self.current_target_position is None or self.calculated_edge_target is None:
            # Hover in place if no target
            cmd_vel.linear.z = (self.transit_height - self.drone_position[2]) * 0.5
            self.velocity_pub.publish(cmd_vel)
            return
        
        # Use the calculated edge target for positioning
        target_position = self.calculated_edge_target
        
        # Calculate vector to the target
        target_vector = target_position[:2] - self.drone_position[:2]
        distance_to_target = np.linalg.norm(target_vector)
        
        # Determine if the drone is in position (horizontally close to target)
        self.is_in_position = distance_to_target <= self.position_threshold
        
        # Calculate desired height based on position
        desired_height = self.operational_height if self.is_in_position else self.transit_height
        
        # If we're already at the minimum distance or closer, don't move horizontally
        if distance_to_target <= self.min_distance:
            # Just hold position
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            self.get_logger().debug(f"At minimum safe distance from edge target: {distance_to_target:.2f}m")
        else:
            # Normalize the vector and set velocity
            if distance_to_target > 0.001:  # Avoid division by zero
                target_vector = target_vector / distance_to_target
                
                # Scale speed based on distance (slower when closer)
                move_speed = min(self.speed, distance_to_target * 0.5)
                
                cmd_vel.linear.x = target_vector[0] * move_speed
                cmd_vel.linear.y = target_vector[1] * move_speed
                
                self.get_logger().debug(f"Moving toward edge target. Distance: {distance_to_target:.2f}m")
        
        # Set vertical movement - handle different behaviors for transit vs. in position
        height_error = desired_height - self.drone_position[2]
        
        # When descending (in position), use a controlled descent rate
        if self.is_in_position and height_error < 0:
            # Descending - use controlled rate
            cmd_vel.linear.z = max(-self.descent_rate, height_error * 0.5)
        else:
            # When ascending or maintaining height, use proportional control
            cmd_vel.linear.z = np.clip(height_error * 0.5, -0.5, 0.8)
        
        # Calculate direction to face - should face toward the current target center
        if self.current_target_position is not None:
            # Vector from drone to current target center (not edge)
            face_vector = self.current_target_position[:2] - self.drone_position[:2]
            if np.linalg.norm(face_vector) > 0.001:
                face_vector = face_vector / np.linalg.norm(face_vector)
                target_yaw = np.arctan2(face_vector[1], face_vector[0])
                yaw_error = self.normalize_angle(target_yaw - self.drone_orientation)
                cmd_vel.angular.z = np.clip(yaw_error * 0.5, 0, 0)
        
        # Publish the command
        self.velocity_pub.publish(cmd_vel)
        
        # Periodically log status (every 5 seconds)
        if hasattr(self, 'last_status_time') and self.get_clock().now().nanoseconds / 1e9 - self.last_status_time < 5.0:
            pass
        else:
            position_status = "in position" if self.is_in_position else "in transit"
            height_status = f"height: {self.drone_position[2]:.1f}m → {desired_height:.1f}m"
            
            # Calculate distance to current target center (for status reporting)
            if self.current_target_position is not None:
                center_dist = np.linalg.norm(self.current_target_position[:2] - self.drone_position[:2])
                dist_info = f"Edge dist: {distance_to_target:.2f}m, Center dist: {center_dist:.2f}m"
            else:
                dist_info = f"Edge dist: {distance_to_target:.2f}m"
            
            status = f"Targeting far edge ({position_status}). {dist_info}, {height_status}"
            if distance_to_target <= self.min_distance:
                status = f"At far edge position ({position_status}). {dist_info}, {height_status}"
            
            self.publish_status(status)
            self.last_status_time = self.get_clock().now().nanoseconds / 1e9
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    """
    Main function for the ROS2 node.
    """
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)
    
    # Create an instance of our node
    node = FixedDirectionWranglerNode()
    
    try:
        # This will block until the node is interrupted (Ctrl-C)
        # During this time, callbacks will be called as data comes in
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl-C gracefully
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        # Always execute this cleanup code when the node stops
        
        # Send stop command before shutting down to ensure drone stops moving
        stop_msg = Twist()
        node.velocity_pub.publish(stop_msg)
        
        # Clean up ROS2 resources
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # This is the entry point when the script is run directly
    main()