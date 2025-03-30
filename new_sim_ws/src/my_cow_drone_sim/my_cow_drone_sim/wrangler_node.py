#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
import time
import math


class WranglerNode(Node):
    """
    Streamlined ROS2 node for autonomous drone control to herd cattle
    based on push positions and directions from the cattle visualizer.
    """
    
    def __init__(self):
        super().__init__('wrangler_node')
        
        # Configuration parameters
        self.transit_height = 3.0     # Height when moving between positions (m)
        self.pushing_height = 0.5     # Height when actively pushing (m)
        self.max_xy_speed = 2.0       # Maximum horizontal speed (m/s)
        self.max_z_speed = 3.0        # Maximum vertical speed (m/s)
        self.max_angular_speed = 1.0  # Maximum rotation speed (rad/s)
        self.position_threshold = 0.1  # Distance to consider position reached (m)
        self.min_follow_distance = 1.0  # Minimum distance behind cattle (m)
        self.max_follow_distance = 5.0  # Maximum distance behind cattle (m)
        self.distance_per_cow = 0.5    # Additional distance per cow in cluster (m)
        self.cluster_timeout = 20.0    # Maximum time to push a cluster (seconds)
        self.goal_position = np.array([7.0, 7.0, 0.0])  # Goal position
        
        # Publishers
        self.velocity_pub = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/wrangler/status', 10)
        
        # State tracking
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.drone_orientation = 0.0
        self.cattle_positions = {}
        
        # Path planning data
        self.push_positions = []      # List of push positions [(id, position), ...]
        self.push_directions = []     # List of push directions [(id, direction), ...]
        self.current_cluster_idx = 0  # Index of the current cluster being pushed
        self.cluster_start_time = time.time()
        self.last_status_time = time.time()
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Watchdog timer
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.stuck_count = 0
        self.watchdog_timer = self.create_timer(5.0, self.watchdog)
        
        self.get_logger().info('Streamlined Wrangler node initialized')
    
    def setup_subscribers(self):
        """Set up all subscribers"""
        # Subscribe to drone odometry
        self.create_subscription(
            Odometry,
            '/drone/odom',
            self.drone_callback,
            10
        )
        
        # Subscribe to cattle odometry
        for i in range(1, 13):  # 12 cattle
            cow_id = f'cow{i}'
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
        
        # Subscribe to visualizer markers
        self.create_subscription(
            MarkerArray,
            '/visualization/markers',
            self.markers_callback,
            10
        )
        
        # Subscribe to optimal path
        self.create_subscription(
            Path,
            '/visualization/optimal_path',
            self.path_callback,
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
    
    def cattle_callback(self, msg, cattle_id):
        """Update cattle position"""
        pos = msg.pose.pose.position
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
    
    def markers_callback(self, msg):
        """Process markers from the visualizer to extract push information"""
        push_positions = []
        push_directions = []
        
        # First, collect all push positions
        for marker in msg.markers:
            if marker.ns == "push_position" and marker.type == Marker.SPHERE:
                pos = np.array([
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z
                ])
                push_positions.append((marker.id, pos))
        
        # Then collect push directions
        for marker in msg.markers:
            if marker.ns == "push_arrow" and marker.type == Marker.ARROW and len(marker.points) >= 2:
                start = np.array([marker.points[0].x, marker.points[0].y])
                end = np.array([marker.points[1].x, marker.points[1].y])
                
                # Calculate and normalize direction vector
                direction = end - start
                distance = np.linalg.norm(direction)
                if distance > 0.001:
                    direction = direction / distance
                    push_directions.append((marker.id, direction))
        
        # Sort by ID (priority from visualizer)
        push_positions.sort(key=lambda x: x[0])
        push_directions.sort(key=lambda x: x[0])
        
        # Update only if we have valid data
        if push_positions and push_directions:
            self.push_positions = push_positions
            self.push_directions = push_directions
    
    def path_callback(self, msg):
        """Process the optimal path from the visualizer"""
        if not msg.poses:
            return
            
        # Extract waypoints (not currently used in streamlined implementation)
        path = []
        for pose in msg.poses:
            pos = pose.pose.position
            path.append(np.array([pos.x, pos.y, pos.z]))
    
    def publish_status(self, message):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def watchdog(self):
        """Watchdog to detect if the drone is stuck"""
        # Check if drone has moved
        distance = np.linalg.norm(self.drone_position[:2] - self.last_position[:2])
        
        if distance < 0.1:
            self.stuck_count += 1
            
            if self.stuck_count >= 3:  # Stuck for ~15 seconds
                self.get_logger().warn('Drone appears stuck, advancing to next cluster')
                self.advance_to_next_cluster()
                self.stuck_count = 0
        else:
            # Reset counter if moving
            self.stuck_count = 0
        
        # Update last position
        self.last_position = self.drone_position.copy()
    
    def get_current_push_data(self):
        """Get the current push position and direction"""
        if not self.push_positions or not self.push_directions:
            return None, None
            
        # Ensure we're within bounds
        if self.current_cluster_idx >= len(self.push_positions):
            self.current_cluster_idx = 0
            
        # Get current push position and direction
        push_position = self.push_positions[self.current_cluster_idx][1]
        
        # Find matching direction
        push_direction = None
        for idx, direction in self.push_directions:
            if idx == self.push_positions[self.current_cluster_idx][0]:
                push_direction = direction
                break
                
        return push_position, push_direction
    
    def calculate_velocity(self, target, speed_limit=None):
        """Calculate velocity to reach the target position"""
        if speed_limit is None:
            speed_limit = self.max_xy_speed
            
        # Get direction vector to target
        direction = target[:2] - self.drone_position[:2]
        distance = np.linalg.norm(direction)
        
        # No movement needed if very close
        if distance < 0.1:
            return np.zeros(2)
            
        # Normalize and scale by distance (slow down as we approach)
        direction = direction / distance
        speed = min(speed_limit, distance)
        
        return direction * speed
    
    def is_position_reached(self, target, threshold=None):
        """Check if target position is reached"""
        if threshold is None:
            threshold = self.position_threshold
            
        distance = np.linalg.norm(self.drone_position[:2] - target[:2])
        return distance < threshold
    
    def calculate_cattle_centroid(self, max_distance=5.0):
        """Calculate centroid of nearby cattle"""
        if not self.cattle_positions:
            return None
            
        # Get push position
        push_position, _ = self.get_current_push_data()
        if push_position is None:
            return None
            
        # Find all cattle within max_distance of push position
        nearby_cattle = []
        for cow_id, cow_pos in self.cattle_positions.items():
            distance = np.linalg.norm(cow_pos[:2] - push_position[:2])
            if distance < max_distance:
                nearby_cattle.append(cow_pos)
                
        # Calculate centroid if we have nearby cattle
        if nearby_cattle:
            return np.mean(nearby_cattle, axis=0)
            
        return None
    
    def get_cluster_size(self, max_distance=5.0):
        """Get the number of cattle in the current cluster"""
        if not self.cattle_positions:
            return 0
            
        push_position, _ = self.get_current_push_data()
        if push_position is None:
            return 0
            
        # Count nearby cattle
        count = 0
        for cow_id, cow_pos in self.cattle_positions.items():
            distance = np.linalg.norm(cow_pos[:2] - push_position[:2])
            if distance < max_distance:
                count += 1
                
        return count
    
    def calculate_dynamic_follow_distance(self):
        """Calculate the dynamic follow distance based on cluster size"""
        cluster_size = self.get_cluster_size()
        
        # Base distance plus additional distance per cow
        distance = self.min_follow_distance + (cluster_size * self.distance_per_cow)
        
        # Clamp to maximum
        return min(distance, self.max_follow_distance)
    
    def calculate_position_behind_cattle(self):
        """Calculate position behind cattle based on push direction and cluster size"""
        # Get cattle centroid and push direction
        centroid = self.calculate_cattle_centroid()
        _, push_direction = self.get_current_push_data()
        
        if centroid is None or push_direction is None:
            return None
        
        # Calculate dynamic follow distance
        follow_distance = self.calculate_dynamic_follow_distance()
        
        # Calculate position behind cattle
        position_behind = centroid[:2] - (push_direction * follow_distance)
        
        # Add transit height for Z coordinate
        return np.array([position_behind[0], position_behind[1], self.transit_height])
    
    def should_switch_clusters(self):
        """Determine if we should switch to the next cluster"""
        # Check for timeout
        elapsed_time = time.time() - self.cluster_start_time
        if elapsed_time > self.cluster_timeout:
            self.publish_status(f"Pushing timeout for cluster {self.current_cluster_idx}")
            return True
            
        # Check if cattle are near goal
        centroid = self.calculate_cattle_centroid()
        if centroid is not None:
            distance_to_goal = np.linalg.norm(centroid[:2] - self.goal_position[:2])
            if distance_to_goal < 2.0:
                self.publish_status(f"Cattle near goal ({distance_to_goal:.1f}m)")
                return True
                
        # No need to switch yet
        return False
    
    def advance_to_next_cluster(self):
        """Move to the next cluster"""
        if not self.push_positions:
            return False
            
        # Advance to next cluster
        self.current_cluster_idx = (self.current_cluster_idx + 1) % len(self.push_positions)
        self.cluster_start_time = time.time()
        self.publish_status(f"Advancing to cluster {self.current_cluster_idx}")
        return True
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def control_loop(self):
        """Main control loop"""
        # Skip if no cattle data yet
        if not self.cattle_positions:
            return
            
        # Get current push position and direction
        push_position, push_direction = self.get_current_push_data()
        if push_position is None:
            return  # No data available yet
        
        # Create velocity message
        cmd_vel = Twist()
        
        # Check the drone's current height
        is_at_transit_height = abs(self.drone_position[2] - self.transit_height) < 0.5
        is_at_pushing_height = abs(self.drone_position[2] - self.pushing_height) < 0.5
        
        # Get cattle centroid and calculate position behind cattle
        cattle_centroid = self.calculate_cattle_centroid()
        position_behind_cattle = self.calculate_position_behind_cattle()
        
        # MODIFIED LOGIC TO POSITION BEHIND CATTLE
        
        # If no cattle are detected yet, use original push position
        target_position = push_position.copy()
        if position_behind_cattle is not None:
            # Use position behind cattle instead
            target_position = position_behind_cattle
        
        # Calculate distance to target position
        distance_to_target = np.linalg.norm(self.drone_position[:2] - target_position[:2])
        
        # 1. If we're not at transit height and not close to target position,
        #    adjust height while moving toward target
        if not is_at_transit_height and distance_to_target > 2.0:
            # Move to transit height
            height_error = self.transit_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
            
            # Always move horizontally toward target position
            velocity = self.calculate_velocity(target_position)
            cmd_vel.linear.x = velocity[0]
            cmd_vel.linear.y = velocity[1]
            
            # Log status periodically
            if time.time() - self.last_status_time > 5.0:
                self.publish_status(f"Moving to position behind cattle while adjusting altitude: {self.drone_position[2]:.1f}m → {self.transit_height}m")
                self.last_status_time = time.time()
        
        # 2. At transit height, move to position behind cattle
        elif is_at_transit_height and distance_to_target > 1.0:
            # Move horizontally to position behind cattle
            velocity = self.calculate_velocity(target_position)
            cmd_vel.linear.x = velocity[0]
            cmd_vel.linear.y = velocity[1]
            
            # Maintain transit height
            height_error = self.transit_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
            
            # Log status periodically
            if time.time() - self.last_status_time > 5.0:
                follow_distance = self.calculate_dynamic_follow_distance()
                cluster_size = self.get_cluster_size()
                self.publish_status(
                    f"Moving behind cluster {self.current_cluster_idx} "
                    f"({cluster_size} cattle): {distance_to_target:.1f}m away, "
                    f"following at {follow_distance:.1f}m"
                )
                self.last_status_time = time.time()
        
        # 3. At position behind cattle but still at transit height, descend for pushing
        elif is_at_transit_height and distance_to_target <= 1.0:
            # Start descending while maintaining position behind cattle
            height_error = self.pushing_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
            
            # Continue tracking position behind cattle
            if cattle_centroid is not None and push_direction is not None:
                # Recalculate ideal position as we descend
                follow_distance = self.calculate_dynamic_follow_distance()
                ideal_position = cattle_centroid[:2] - push_direction * follow_distance
                
                # Move toward ideal position
                velocity = self.calculate_velocity(np.append(ideal_position, 0), speed_limit=0.8)
                cmd_vel.linear.x = velocity[0]
                cmd_vel.linear.y = velocity[1]
            else:
                # Maintain position with small adjustments if no cattle data
                velocity = self.calculate_velocity(target_position, speed_limit=0.5)
                cmd_vel.linear.x = velocity[0]
                cmd_vel.linear.y = velocity[1]
            
            # Orient toward push direction
            if push_direction is not None:
                target_yaw = np.arctan2(push_direction[1], push_direction[0])
                yaw_error = self.normalize_angle(target_yaw - self.drone_orientation)
                cmd_vel.angular.z = np.clip(yaw_error, -self.max_angular_speed, self.max_angular_speed)
            
            # Log status periodically
            if time.time() - self.last_status_time > 5.0:
                self.publish_status(f"Descending for pushing while tracking cattle: {self.drone_position[2]:.1f}m → {self.pushing_height}m")
                self.last_status_time = time.time()
        
        # 4. At pushing height, perform herding behavior
        elif is_at_pushing_height:
            # Check if we should switch clusters
            if self.should_switch_clusters():
                # Ascend before moving to next cluster
                height_error = self.transit_height - self.drone_position[2]
                cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
                
                # Continue moving horizontally while ascending
                if position_behind_cattle is not None:
                    velocity = self.calculate_velocity(position_behind_cattle, speed_limit=0.8)
                    cmd_vel.linear.x = velocity[0]
                    cmd_vel.linear.y = velocity[1]
                
                # If we're getting close to transit height, advance to next cluster
                if height_error < 1.0:
                    self.advance_to_next_cluster()
            else:
                # Normal pushing behavior
                if cattle_centroid is not None and push_direction is not None:
                    # Calculate dynamic follow distance
                    follow_distance = self.calculate_dynamic_follow_distance()
                    
                    # Calculate ideal position behind cattle
                    ideal_position = cattle_centroid[:2] - push_direction * follow_distance
                    
                    # Calculate direction to ideal position
                    to_ideal = ideal_position - self.drone_position[:2]
                    distance_to_ideal = np.linalg.norm(to_ideal)
                    
                    # If too far from ideal position, move towards it
                    if distance_to_ideal > 1.0:
                        velocity = self.calculate_velocity(np.append(ideal_position, 0), speed_limit=1.0)
                        cmd_vel.linear.x = velocity[0]
                        cmd_vel.linear.y = velocity[1]
                    else:
                        # Close enough, add gentle push in push direction
                        cmd_vel.linear.x = push_direction[0] * 0.5
                        cmd_vel.linear.y = push_direction[1] * 0.5
                    
                    # Orient towards cattle
                    to_cattle = cattle_centroid[:2] - self.drone_position[:2]
                    if np.linalg.norm(to_cattle) > 0.1:
                        target_yaw = np.arctan2(to_cattle[1], to_cattle[0])
                        yaw_error = self.normalize_angle(target_yaw - self.drone_orientation)
                        cmd_vel.angular.z = np.clip(yaw_error, -self.max_angular_speed, self.max_angular_speed)
                    
                    # Maintain pushing height
                    height_error = self.pushing_height - self.drone_position[2]
                    cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
                    
                    # Log status periodically
                    if time.time() - self.last_status_time > 5.0:
                        elapsed = time.time() - self.cluster_start_time
                        remaining = max(0, self.cluster_timeout - elapsed)
                        cluster_size = self.get_cluster_size()
                        self.publish_status(
                            f"Pushing cluster {self.current_cluster_idx} "
                            f"({cluster_size} cattle): "
                            f"following at {follow_distance:.1f}m, "
                            f"time: {elapsed:.0f}s/{remaining:.0f}s"
                        )
                        self.last_status_time = time.time()
                else:
                    # No cattle found or no push direction, ascend and move to next cluster
                    height_error = self.transit_height - self.drone_position[2]
                    cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
                    
                    # Move horizontally to the next push position while ascending
                    next_idx = (self.current_cluster_idx + 1) % len(self.push_positions)
                    if next_idx < len(self.push_positions):
                        next_position = self.push_positions[next_idx][1]
                        velocity = self.calculate_velocity(next_position, speed_limit=1.0)
                        cmd_vel.linear.x = velocity[0]
                        cmd_vel.linear.y = velocity[1]
                    
                    if height_error < 1.0:  # Getting close to transit height
                        self.advance_to_next_cluster()
        
        # 5. In transition between heights, handle smoothly with simultaneous movement
        else:
            # Determine if we're ascending or descending
            if distance_to_target <= 1.0:  # Near target position, should be descending
                target_height = self.pushing_height
            else:  # Far from target position, should be at transit height
                target_height = self.transit_height
            
            # Move toward target height
            height_error = target_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -self.max_z_speed, self.max_z_speed)
            
            # Always move horizontally, adjust speed based on distance
            speed_limit = 1.0 if distance_to_target > 2.0 else 0.5
            
            # If we have cattle data, track position behind them
            if cattle_centroid is not None and push_direction is not None:
                follow_distance = self.calculate_dynamic_follow_distance()
                ideal_position = cattle_centroid[:2] - push_direction * follow_distance
                
                velocity = self.calculate_velocity(np.append(ideal_position, 0), speed_limit=speed_limit)
                cmd_vel.linear.x = velocity[0]
                cmd_vel.linear.y = velocity[1]
            else:
                # Use target position if no cattle data
                velocity = self.calculate_velocity(target_position, speed_limit=speed_limit)
                cmd_vel.linear.x = velocity[0]
                cmd_vel.linear.y = velocity[1]
        
        # Publish velocity command
        self.velocity_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = WranglerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        # Send stop command
        stop_msg = Twist()
        node.velocity_pub.publish(stop_msg)
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()