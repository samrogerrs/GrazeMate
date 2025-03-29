#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import time
import math
from visualization_msgs.msg import MarkerArray

class WranglerNode(Node):
    """
    Enhanced ROS2 node for drone movement that follows optimal paths
    from the cattle visualizer to efficiently herd cattle towards a goal.
    """
    
    def __init__(self):
        super().__init__('wrangler_node')
        
        # Create publisher for velocity commands
        self.publisher = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        
        # Subscribe to drone odometry
        self.drone_sub = self.create_subscription(
            Odometry,
            '/drone/odom',
            self.drone_callback,
            10)
            
        # Subscribe to cattle odometry
        self.cattle_positions = {}
        self.cattle_velocities = {}
        
        # You'll need to adjust this to match your cattle count
        self.cattle_count = 12  # Assuming 12 cattle
        for i in range(1, self.cattle_count + 1):
            cow_id = f'cow{i}'
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10)
        
        # Subscribe to visualizer paths
        self.optimal_path_sub = self.create_subscription(
            Path,
            '/visualization/optimal_path',
            self.optimal_path_callback,
            10)
            
        # Subscribe to marker array for push positions and directions
        self.markers_sub = self.create_subscription(
            MarkerArray,
            '/visualization/markers',
            self.markers_callback,
            10)
        
        # Path following parameters
        self.path_waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_reached_threshold = 0.8  # meters
        self.lookahead_distance = 1.5  # meters for path following
        
        # Herding parameters
        self.goal_position = np.array([7.0, 7.0, 0.0])  # Goal position at (7,7)
        self.speed = 3  # Drone speed in m/s
        
        # Altitude control parameters
        self.engagement_altitude = 0.5  # Height when actively herding cattle
        self.transit_altitude = 3.5  # Height when moving to position without engaging
        self.current_target_altitude = self.transit_altitude  # Default to transit altitude
        self.altitude_transition_rate = 2.0  # Rate of altitude change (m/s)
        
        # Engagement parameters
        self.engagement_distance = 4.0  # Distance threshold to consider engaging with cattle
        self.is_engaging = False  # Track whether currently engaging with cattle
        self.engagement_cooldown = 0.0  # Cooldown timer to prevent rapid altitude changes
        
        # Push position tracking
        self.push_positions = []  # List of push positions from markers
        self.current_push_target = None  # Current push position being targeted
        self.current_push_direction = None  # Direction to push from current target
        
        # Drone state
        self.drone_position = np.array([0.0, 0.0, 1.0])  # Default starting position
        self.drone_velocity = np.array([0.0, 0.0, 0.0])  # Current velocity
        self.drone_yaw = 0.0  # Yaw orientation
        self.start_time = time.time()
        self.last_movement_time = time.time()  # Track last movement time
        
        # Mode tracking
        self.operation_mode = "TRANSIT"  # TRANSIT, POSITIONING, PUSHING
        
        # Timer for control
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Watchdog timer to detect if the drone is stuck
        self.watchdog_timer = self.create_timer(5.0, self.watchdog_callback)
        self.last_position = np.array([0.0, 0.0, 0.0])
        
        self.get_logger().info('Path Following Wrangler node initialized')
    
    def drone_callback(self, msg):
        """
        Update drone position and velocity from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.drone_position = np.array([pos.x, pos.y, pos.z])
        self.drone_velocity = np.array([vel.x, vel.y, vel.z])
        
        # Extract yaw from quaternion (assuming drone only rotates around z-axis)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions and velocities from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        self.cattle_velocities[cattle_id] = np.array([vel.x, vel.y, vel.z])
    
    def optimal_path_callback(self, msg):
        """
        Process optimal path from visualizer
        """
        if not msg.poses:
            return
            
        # Extract waypoints from path message
        self.path_waypoints = []
        for pose in msg.poses:
            pos = pose.pose.position
            self.path_waypoints.append(np.array([pos.x, pos.y, pos.z]))
        
        # Reset waypoint tracking if we get a new path
        self.current_waypoint_idx = 0
        
        # Log path update
        self.get_logger().info(f'Received new path with {len(self.path_waypoints)} waypoints')
    
    def markers_callback(self, msg):
        """
        Process visualizer markers to extract push positions and directions
        """
        # Extract push positions from markers
        new_push_positions = []
        
        for marker in msg.markers:
            if marker.ns == "push_position" and marker.type == 1:  # SPHERE type
                # Extract position
                pos = marker.pose.position
                push_pos = np.array([pos.x, pos.y, pos.z])
                
                # Add to list with marker ID for ordering
                new_push_positions.append((marker.id, push_pos))
        
        # Sort by ID (priority order from visualizer)
        if new_push_positions:
            new_push_positions.sort(key=lambda x: x[0])
            self.push_positions = [pos for _, pos in new_push_positions]
            
            # Update current target if we don't have one yet
            if self.current_push_target is None and self.push_positions:
                self.current_push_target = self.push_positions[0]
                # Look for the corresponding push arrow direction
                self.find_push_direction_for_target(msg, 0)
    
    def find_push_direction_for_target(self, marker_array, target_idx):
        """
        Find the push direction for a specific push position target
        """
        for marker in marker_array.markers:
            if marker.ns == "push_arrow" and marker.id == target_idx and marker.type == 0:  # ARROW type
                if len(marker.points) >= 2:
                    # Extract direction from arrow start and end points
                    start = marker.points[0]
                    end = marker.points[1]
                    
                    # Calculate direction vector
                    direction = np.array([end.x - start.x, end.y - start.y])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        self.current_push_direction = direction
                        return True
        
        # If no direction found, use a default
        self.current_push_direction = np.array([1.0, 0.0])  # Default direction
        return False
    
    def watchdog_callback(self):
        """
        Check if the drone has moved in the last 5 seconds
        """
        if np.linalg.norm(self.drone_position[:2] - self.last_position[:2]) < 0.1:
            self.get_logger().warn('Drone appears to be stuck! Sending new movement command.')
            # Send a small jog command to unstick the drone
            msg = Twist()
            msg.linear.x = 0.5
            msg.linear.y = 0.5
            # Maintain target altitude
            z_error = self.current_target_altitude - self.drone_position[2]
            msg.linear.z = np.clip(z_error, -0.5, 0.5)
            self.publisher.publish(msg)
            
            # Also advance to next waypoint if we're stuck on the current one
            if self.path_waypoints and self.current_waypoint_idx < len(self.path_waypoints) - 1:
                self.current_waypoint_idx += 1
                self.get_logger().info(f'Skipping to next waypoint: {self.current_waypoint_idx}')
        
        # Update last position
        self.last_position = self.drone_position.copy()
    
    def get_closest_cattle(self, position, max_distance=5.0):
        """
        Find the closest cattle to a given position within max_distance
        """
        closest_cattle = None
        min_distance = float('inf')
        
        for cattle_id, cattle_pos in self.cattle_positions.items():
            distance = np.linalg.norm(cattle_pos[:2] - position[:2])
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_cattle = (cattle_id, cattle_pos)
        
        return closest_cattle
    
    
    def set_operation_mode(self):
        """
        Simplified function to determine the current operation mode based on position 
        and proximity to cattle
        """
        # Default to transit mode
        self.operation_mode = "TRANSIT"
        self.current_target_altitude = self.transit_altitude
        
        # If no path available yet, stay in transit mode
        if not self.path_waypoints:
            return
        
        # Check for any cattle nearby (within engagement distance)
        nearest_cattle = None
        min_distance = float('inf')
        
        for cattle_id, cattle_pos in self.cattle_positions.items():
            distance = np.linalg.norm(self.drone_position[:2] - cattle_pos[:2])
            if distance < min_distance:
                min_distance = distance
                nearest_cattle = (cattle_id, cattle_pos)
        
        # If we're close to cattle, determine if we should be pushing
        if nearest_cattle and min_distance < self.engagement_distance:
            cattle_id, cattle_pos = nearest_cattle
            
            # Check if cattle needs to be pushed (not already at goal)
            cattle_to_goal = np.linalg.norm(self.goal_position[:2] - cattle_pos[:2])
            
            if cattle_to_goal > 2.0:  # Only push if cattle isn't already near goal
                # Simplified engagement logic - if we're close to cattle, we're pushing
                self.operation_mode = "PUSHING"
                self.current_target_altitude = self.engagement_altitude
                
                # Update current push target to be this cattle
                self.current_push_target = cattle_pos
                
                # Calculate push direction (toward goal)
                cattle_to_goal_vec = self.goal_position[:2] - cattle_pos[:2]
                if np.linalg.norm(cattle_to_goal_vec) > 0:
                    self.current_push_direction = cattle_to_goal_vec / np.linalg.norm(cattle_to_goal_vec)
                
                # Log that we're pushing
                self.get_logger().info(f'PUSHING cattle {cattle_id}, distance to goal: {cattle_to_goal:.2f}m')
                return
        
        # If we get here, we're not pushing - stay in transit mode following waypoints
        # This means we'll follow the path at a higher altitude

    
    def get_next_waypoint(self):
        """
        Get the next waypoint in the path, using lookahead for smooth following
        """
        if not self.path_waypoints:
            return None
            
        # Start with current waypoint
        current_idx = self.current_waypoint_idx
        
        # Mark current waypoint as reached if close enough
        if current_idx < len(self.path_waypoints):
            current_waypoint = self.path_waypoints[current_idx]
            distance = np.linalg.norm(self.drone_position[:2] - current_waypoint[:2])
            
            if distance < self.waypoint_reached_threshold:
                # Advance to next waypoint
                self.current_waypoint_idx = min(current_idx + 1, len(self.path_waypoints) - 1)
                current_idx = self.current_waypoint_idx
                
                # If this is a push position waypoint, update the push target
                if self.push_positions and current_idx < len(self.push_positions):
                    self.current_push_target = self.push_positions[current_idx]
                    
                    # We'll update the push direction dynamically when we're actually pushing
                    # This is just to initialize the direction
                    self.current_push_direction = np.array([1.0, 0.0])  # Default direction
        
        # Check if we need lookahead
        
        if current_idx < len(self.path_waypoints) - 1:
            current_waypoint = self.path_waypoints[current_idx]
            next_waypoint = self.path_waypoints[current_idx + 1]
            
            # Calculate distance to current waypoint
            distance_to_current = np.linalg.norm(self.drone_position[:2] + current_waypoint[:2])
            
            # If we're close enough to current waypoint, start moving toward next one
            if distance_to_current < self.lookahead_distance:
                # Blend between current and next waypoints based on distance
                blend_factor = 1.0 - (distance_to_current / self.lookahead_distance)
                blend_factor = max(0.0, min(1.0, blend_factor))  # Clamp between 0 and 1
                
                # Interpolate between current and next waypoint
                target = current_waypoint[:2] * (1.0 - blend_factor) + next_waypoint[:2] * blend_factor
                return target
        
        # Return current waypoint if we have one
        if current_idx < len(self.path_waypoints):
            return self.path_waypoints[current_idx][:2]
            
        # Fallback to goal position if we have no valid waypoints
        return self.goal_position[:2]

    def calculate_push_position(self, cattle_pos):
        """
        Calculate position for pushing cattle towards goal, ensuring drone approaches
        from the correct direction (2.75m behind cattle in push direction)
        """
        # If we have a current_push_direction from markers, use that
        if self.current_push_direction is not None:
            # This is the direction we want cattle to move (toward goal)
            push_direction = self.current_push_direction
        else:
            # Calculate push direction based on cattle-to-goal vector as fallback
            cattle_to_goal = self.goal_position[:2] - cattle_pos[:2]
            distance_to_goal = np.linalg.norm(cattle_to_goal)
            
            if distance_to_goal > 0.1:
                push_direction = cattle_to_goal / distance_to_goal
            else:
                push_direction = np.array([1.0, 0.0])  # Default direction
        
        # Position behind cattle - exactly 2.75m behind in the push direction
        # Use NEGATIVE push_direction to go in the opposite direction from the goal
        behind_distance = 1.00  # meters, as specified
        push_position = cattle_pos[:2] - (push_direction * behind_distance)
        
        # Perpendicular vector for oscillation (if needed)
        perpendicular = np.array([-push_direction[1], push_direction[0]])
        
        # Apply oscillation based on time (reduced amplitude)
        amplitude = 0.5  # meters (reduced from 1.5)
        frequency = 0.3  # Hz
        
        current_time = time.time()
        oscillation = amplitude * math.sin(2 * math.pi * frequency * (current_time - self.start_time))
        
        # Add small perpendicular oscillation (can be disabled by setting amplitude to 0)
        push_position = push_position + (perpendicular * oscillation)
        
        return push_position
        
    def altitude_control(self, msg, dt):
        """
        Dynamic altitude control to maintain target height with smooth transitions
        """
        # Current altitude
        current_z = self.drone_position[2]
        
        # Calculate error
        z_error = self.current_target_altitude - current_z
        
        # Adjust altitude with rate limiting for smooth transitions
        if abs(z_error) > 0.1:  # Only adjust if error is significant
            # Apply proportional control with rate limiting
            desired_z_velocity = np.clip(z_error, -self.altitude_transition_rate, self.altitude_transition_rate)
            msg.linear.z = desired_z_velocity
        else:
            # Maintain altitude with small corrections
            msg.linear.z = np.clip(z_error * 2.0, -0.5, 0.5)
        
        return msg
    
    
    def timer_callback(self):
        """
        Simplified main control loop for path following and herding
        """
        # Skip if no cattle data received yet
        if not self.cattle_positions:
            self.get_logger().info('No cattle data received yet.')
            return
        
        # Update time tracking
        current_time = time.time()
        dt = current_time - self.last_movement_time
        self.last_movement_time = current_time
        
        # Set operation mode based on current state
        self.set_operation_mode()
        
        # Calculate target position based on mode
        if self.operation_mode == "PUSHING" and self.current_push_target is not None:
            # We're in pushing mode with a valid target
            cattle_pos = self.current_push_target
            
            # Calculate ideal push position (behind cattle)
            target_position = self.calculate_push_position(cattle_pos)
            
            # Log pushing information
            if int(current_time) % 3 == 0:
                self.get_logger().info(f'Push position: {target_position}, Drone: {self.drone_position[:2]}')
        else:
            # We're in transit mode, follow path
            target_position = self.get_next_waypoint()
        
        # Create velocity message
        msg = Twist()
        
        if target_position is not None:
            # Direction vector to target
            direction_to_target = target_position - self.drone_position[:2]
            distance = np.linalg.norm(direction_to_target)
            
            if distance > 0.1:  # Only move if we're not already at the target
                # Normalize direction
                direction_normalized = direction_to_target / distance
                
                # Scale velocity based on distance (smoother approach)
                # Use higher speed when in transit mode, slower when positioning for push
                if self.operation_mode == "PUSHING" and distance < 3.0:
                    # Slow approach when getting into push position
                    max_speed = self.speed * 0.7
                else:
                    # Normal transit speed
                    max_speed = self.speed * 1.3
                    
                speed_factor = min(1.0, distance / 2.0)
                velocity_scale = max_speed * speed_factor
                
                # Set velocity components
                msg.linear.x = direction_normalized[0] * velocity_scale
                msg.linear.y = direction_normalized[1] * velocity_scale
                
                # Set angular velocity to align drone with movement direction
                target_yaw = math.atan2(direction_normalized[1], direction_normalized[0])
                yaw_error = self.normalize_angle(target_yaw - self.drone_yaw)
                msg.angular.z = np.clip(yaw_error * 1.0, -1.0, 1.0)  # P controller for rotation
        
        # Apply altitude control
        msg = self.altitude_control(msg, dt)
        
        # Publish velocity command
        self.publisher.publish(msg)
        
        # Log status occasionally
        if int(current_time) % 5 == 0 and int(current_time) != int(self.start_time):
            self.get_logger().info(
                f'Mode: {self.operation_mode}, Altitude: {self.drone_position[2]:.1f}m (Target: {self.current_target_altitude:.1f}m)'
            )
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    wrangler_node = WranglerNode()
    
    try:
        rclpy.spin(wrangler_node)
    except KeyboardInterrupt:
        wrangler_node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        # Stop the drone before shutting down
        stop_msg = Twist()
        wrangler_node.publisher.publish(stop_msg)
        wrangler_node.get_logger().info('Stopping drone and shutting down node')
        
        # Clean up
        wrangler_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()