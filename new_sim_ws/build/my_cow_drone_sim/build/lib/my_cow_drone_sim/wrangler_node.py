#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
import time
import math


class WranglerNode(Node):
    """
    Simplified ROS2 node for autonomous drone control to herd cattle,
    inspired by the 2D implementation but adapted for 3D.
    """
    
    def __init__(self):
        super().__init__('wrangler_node')
        
        # Configuration parameters
        self.transit_height = 3.0     # Height when moving between positions (m)
        self.pushing_height = 0.5     # Height when actively pushing (m)
        self.speed = 2.0              # Base movement speed (m/s)
        self.min_distance = 3.0       # Minimum distance to maintain from cattle (m)
        self.oscillation_amplitude = 1.0      # Maximum horizontal offset (m)
        self.oscillation_frequency = 0.5      # Oscillation frequency (Hz)
        self.goal_position = np.array([7.0, 7.0, 0.0])  # Goal position
        
        # Publishers
        self.velocity_pub = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/wrangler/status', 10)
        
        # State tracking
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.drone_orientation = 0.0
        self.cattle_positions = {}
        self.cattle_velocities = {}
        self.fly_high = False
        self.last_status_time = time.time()
        self.time_offset = time.time()  # For oscillation calculation
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Simplified Wrangler node initialized')
    
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
        
        # Subscribe to visualizer markers (optional, for push direction info)
        self.create_subscription(
            MarkerArray,
            '/visualization/markers',
            self.markers_callback,
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
        """Update cattle position and velocity"""
        pos = msg.pose.pose.position
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        
        # Extract and store velocity for each cow
        vel = msg.twist.twist.linear
        self.cattle_velocities[cattle_id] = np.array([vel.x, vel.y, vel.z])
    
    def markers_callback(self, msg):
        """Process markers (simplified - we'll mainly use it for debugging)"""
        # Simplified marker handling - could expand if needed
        for marker in msg.markers:
            if marker.ns == "push_arrow" and marker.type == Marker.ARROW and len(marker.points) >= 2:
                # Log marker info for debugging
                self.get_logger().debug(f"Received push arrow marker: {marker.id}")
    
    def publish_status(self, message):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def get_active_cattle(self, max_distance=8.0):
        """Get cattle that are currently being herded (within max_distance of drone)"""
        if not self.cattle_positions:
            return []
            
        # Find all cattle within max_distance of drone
        active_cattle = []
        for cow_id, cow_pos in self.cattle_positions.items():
            distance = np.linalg.norm(cow_pos[:2] - self.drone_position[:2])
            if distance < max_distance:
                active_cattle.append((cow_id, cow_pos, self.cattle_velocities.get(cow_id, np.zeros(3))))
                
        return active_cattle
    
    def calculate_centroid(self, cattle_list):
        """Calculate centroid of cattle positions"""
        if not cattle_list:
            return None
            
        positions = [pos for _, pos, _ in cattle_list]
        return np.mean(positions, axis=0)
    
    def find_farthest_from_goal(self, cattle_list):
        """Find the cow that's farthest from the goal"""
        if not cattle_list:
            return None, None, None
            
        farthest_id = None
        farthest_pos = None
        farthest_vel = None
        max_distance = -1
        
        for cow_id, cow_pos, cow_vel in cattle_list:
            distance = np.linalg.norm(cow_pos[:2] - self.goal_position[:2])
            if distance > max_distance:
                max_distance = distance
                farthest_id = cow_id
                farthest_pos = cow_pos
                farthest_vel = cow_vel
                
        return farthest_id, farthest_pos, farthest_vel
    
    def should_fly_high(self, active_cattle):
        """Determine if the drone should fly high (equivalent to the 2D class fly_over)"""
        if not active_cattle:
            return False
            
        # Check if drone is too close to any cow
        for _, cow_pos, _ in active_cattle:
            if np.linalg.norm(self.drone_position[:2] - cow_pos[:2]) < self.min_distance:
                return True
                
        # Check if cattle are moving toward goal at sufficient speed
        centroid = self.calculate_centroid(active_cattle)
        if centroid is None:
            return False
            
        # Calculate average velocity of active cattle
        avg_velocity = np.mean([vel[:2] for _, _, vel in active_cattle], axis=0)
        speed = np.linalg.norm(avg_velocity)
        
        # Direction to goal from centroid
        goal_direction = self.goal_position[:2] - centroid[:2]
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
            
        # If moving fast enough in the right direction, fly high
        if speed > 1.0 and np.dot(avg_velocity, goal_direction) > 0.7:
            return True
            
        return False
    
    def calculate_approach_vector(self, farthest_pos, active_centroid):
        """Calculate the best approach vector for herding"""
        if farthest_pos is None or active_centroid is None:
            return np.zeros(2)
            
        # Direction from farthest cow to goal
        direction_to_goal = self.goal_position[:2] - farthest_pos[:2]
        if np.linalg.norm(direction_to_goal) > 0:
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
            
        # Direction from farthest cow to centroid (herd)
        direction_to_centroid = active_centroid[:2] - farthest_pos[:2]
        if np.linalg.norm(direction_to_centroid) > 0:
            direction_to_centroid = direction_to_centroid / np.linalg.norm(direction_to_centroid)
            
        # Combined approach direction (like in the 2D version)
        approach_direction = direction_to_goal + direction_to_centroid
        if np.linalg.norm(approach_direction) > 0:
            approach_direction = approach_direction / np.linalg.norm(approach_direction)
            
        return approach_direction
    
    def apply_oscillation(self, base_position):
        """Apply horizontal oscillation to the position (similar to 2D version)"""
        if base_position is None:
            return None
            
        # Calculate sine oscillation based on time
        time_val = time.time() - self.time_offset
        oscillation = self.oscillation_amplitude * math.sin(time_val * self.oscillation_frequency * 2 * math.pi)
        
        # Apply oscillation perpendicular to the approach direction
        # We'll apply it to the x-coordinate for simplicity
        position = base_position.copy()
        position[0] += oscillation
        
        return position
    
    def control_loop(self):
        """Main control loop"""
        # Skip if no cattle data yet
        if not self.cattle_positions:
            return
            
        # Get active cattle
        active_cattle = self.get_active_cattle()
        if not active_cattle:
            self.publish_status("No active cattle detected")
            return
            
        # Calculate centroid of active cattle
        active_centroid = self.calculate_centroid(active_cattle)
        
        # Find the cow that's farthest from the goal
        farthest_id, farthest_pos, farthest_vel = self.find_farthest_from_goal(active_cattle)
        
        # Determine if we should fly high
        self.fly_high = self.should_fly_high(active_cattle)
        
        # Create velocity message
        cmd_vel = Twist()
        
        # If we're flying high, go to transit height
        if self.fly_high:
            # Move to transit height
            height_error = self.transit_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -2.0, 2.0)
            
            # Log status periodically
            if time.time() - self.last_status_time > 5.0:
                self.publish_status(f"Flying high: {self.drone_position[2]:.1f}m → {self.transit_height}m")
                self.last_status_time = time.time()
        else:
            # Normal herding behavior
            
            # Calculate approach direction
            approach_vector = self.calculate_approach_vector(farthest_pos, active_centroid)
            
            # Calculate base desired position (behind the farthest cow)
            base_position = farthest_pos[:2] - approach_vector * 2.0  # 2m behind the cow
            
            # Apply oscillation for more natural movement
            desired_position_2d = self.apply_oscillation(base_position)
            
            # Set desired height based on distance to the target position
            distance_to_desired = np.linalg.norm(self.drone_position[:2] - desired_position_2d)
            desired_height = self.transit_height if distance_to_desired > 3.0 else self.pushing_height
            
            # Move toward desired position in horizontal plane
            direction_to_desired = desired_position_2d - self.drone_position[:2]
            if np.linalg.norm(direction_to_desired) > 0.1:
                direction_to_desired = direction_to_desired / np.linalg.norm(direction_to_desired)
                move_speed = min(self.speed, distance_to_desired)
                
                cmd_vel.linear.x = direction_to_desired[0] * move_speed
                cmd_vel.linear.y = direction_to_desired[1] * move_speed
            
            # Adjust height
            height_error = desired_height - self.drone_position[2]
            cmd_vel.linear.z = np.clip(height_error, -1.0, 1.0)
            
            # Orient toward the farthest cow
            direction_to_cow = farthest_pos[:2] - self.drone_position[:2]
            if np.linalg.norm(direction_to_cow) > 0.1:
                target_yaw = np.arctan2(direction_to_cow[1], direction_to_cow[0])
                yaw_error = self.normalize_angle(target_yaw - self.drone_orientation)
                cmd_vel.angular.z = np.clip(yaw_error, -1.0, 1.0)
            
            # Log status periodically
            if time.time() - self.last_status_time > 5.0:
                cow_dist = np.linalg.norm(self.drone_position[:2] - farthest_pos[:2])
                self.publish_status(
                    f"Herding cow {farthest_id} at distance {cow_dist:.1f}m, "
                    f"height: {self.drone_position[2]:.1f}m → {desired_height:.1f}m"
                )
                self.last_status_time = time.time()
        
        # Publish velocity command
        self.velocity_pub.publish(cmd_vel)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


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