#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import time
import math

class EnhancedWranglerNode(Node):
    """
    Enhanced ROS2 node for drone movement to herd cattle towards a goal.
    Features dynamic altitude control based on engagement status.
    """
    
    def __init__(self):
        super().__init__('enhanced_wrangler_node')
        
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
        for i in range(self.cattle_count):
            self.create_subscription(
                Odometry,
                f'/cow{i}/odom',
                lambda msg, idx=i: self.cattle_callback(msg, idx),
                10)
        
        # Herding parameters
        self.goal_position = np.array([7.0, 7.0, 0.0])  # Goal position at (7,7)
        self.speed = 1.5  # Drone speed in m/s
        
        # Altitude control parameters
        self.engagement_altitude = 1.0  # Height when actively herding cattle
        self.transit_altitude = 3.0  # Height when moving to position without engaging
        self.current_target_altitude = self.transit_altitude  # Default to transit altitude
        self.altitude_transition_rate = 2  # Rate of altitude change (m/s)
        
        # Engagement parameters
        self.engagement_distance = 4.0  # Distance threshold to consider engaging with cattle
        self.is_engaging = False  # Track whether currently engaging with cattle
        self.engagement_cooldown = 0.0  # Cooldown timer to prevent rapid altitude changes
        
        # Stress distance (to avoid getting too close to cattle)
        self.stress_distance = 1.5  # Distance at which cattle get stressed
        
        # Oscillation parameters
        self.oscillation_amplitude = 1.5  # Maximum distance to oscillate (meters)
        self.oscillation_frequency = 0.4  # Oscillations per second
        self.oscillation_phase = 0.0  # Current phase of oscillation
        
        # Drone state
        self.drone_position = np.array([0.0, 0.0, 1.0])  # Default starting position
        self.drone_velocity = np.array([0.0, 0.0, 0.0])  # Current velocity
        self.start_time = time.time()
        self.last_movement_time = time.time()  # Track last movement time
        
        # Timer for control
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Watchdog timer to detect if the drone is stuck
        self.watchdog_timer = self.create_timer(5.0, self.watchdog_callback)
        self.last_position = np.array([0.0, 0.0, 0.0])
        
        self.get_logger().info('Enhanced Wrangler node initialized with dynamic altitude control')
    
    def drone_callback(self, msg):
        """
        Update drone position and velocity from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.drone_position = np.array([pos.x, pos.y, pos.z])
        self.drone_velocity = np.array([vel.x, vel.y, vel.z])
        self.get_logger().debug(f'Drone odometry received: x={pos.x}, y={pos.y}, z={pos.z}')
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions and velocities from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        self.cattle_velocities[cattle_id] = np.array([vel.x, vel.y, vel.z])
    
    def find_furthest_cow(self):
        """Find the cow that is furthest from the goal"""
        max_distance = -1
        furthest_cow_id = None
        
        for cow_id, position in self.cattle_positions.items():
            distance = np.linalg.norm(position[:2] - self.goal_position[:2])
            if distance > max_distance:
                max_distance = distance
                furthest_cow_id = cow_id
                
        if furthest_cow_id is not None:
            self.get_logger().debug(f"Selected furthest cow: {furthest_cow_id} at distance {max_distance:.2f}")
            
        return furthest_cow_id
    
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
        
        # Update last position
        self.last_position = self.drone_position.copy()
    
    def check_engagement_status(self, furthest_cow_pos):
        """
        Determine if the drone should be engaging with cattle
        and update the target altitude accordingly
        """
        current_time = time.time()
        
        # Calculate 2D distance to the target cow
        distance_to_cow = np.linalg.norm(self.drone_position[:2] - furthest_cow_pos[:2])
        
        # Calculate vector from cow to goal
        cow_to_goal = self.goal_position[:2] - furthest_cow_pos[:2]
        distance_to_goal = np.linalg.norm(cow_to_goal)
        
        # If cow is close to goal, stay in transit mode
        if distance_to_goal < 1.5:
            return False
        
        # Check if we're properly positioned behind the cow
        if distance_to_goal > 0.1:
            cow_to_goal_norm = cow_to_goal / distance_to_goal
        else:
            cow_to_goal_norm = np.array([1.0, 0.0])
        
        # Vector from cow to drone
        cow_to_drone = self.drone_position[:2] - furthest_cow_pos[:2]
        distance_cow_to_drone = np.linalg.norm(cow_to_drone)
        
        # Skip if too far away
        if distance_cow_to_drone < 0.1:
            return False
        
        # Check if we're in the right quadrant (behind the cow relative to goal)
        cow_to_drone_norm = cow_to_drone / distance_cow_to_drone
        dot_product = np.dot(cow_to_goal_norm, cow_to_drone_norm)
        
        # We're properly positioned if the dot product is negative (opposite directions)
        # and we're within engagement distance
        proper_position = dot_product < -0.3  # At least somewhat behind
        proper_distance = 1.5 < distance_cow_to_drone < self.engagement_distance
        
        # We should engage if we're in proper position and distance
        should_engage = proper_position and proper_distance
        
        # Add some debug info
        self.get_logger().debug(f"Position check: distance={distance_cow_to_drone:.2f}, " + 
                            f"dot_product={dot_product:.2f}, should_engage={should_engage}")
        
        # Add cooldown to prevent rapid altitude changes
        if should_engage != self.is_engaging and current_time > self.engagement_cooldown:
            self.is_engaging = should_engage
            self.engagement_cooldown = current_time + 3.0  # 3-second cooldown
            
            if self.is_engaging:
                self.get_logger().info(f'Engaging cattle - descending to {self.engagement_altitude}m')
                self.current_target_altitude = self.engagement_altitude
            else:
                self.get_logger().info(f'Disengaging cattle - ascending to {self.transit_altitude}m')
                self.current_target_altitude = self.transit_altitude
        
        return self.is_engaging
    
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
            
            # Log significant altitude transitions
            if abs(z_error) > 0.5:
                self.get_logger().info(f'Altitude adjustment: current={current_z:.2f}m, ' +
                                    f'target={self.current_target_altitude}m, z_vel={desired_z_velocity:.2f}')
        else:
            # Maintain altitude with small corrections
            msg.linear.z = np.clip(z_error * 2.0, -0.5, 0.5)
        
        return msg
    
    def apply_oscillation(self, base_position, direction_to_goal):
        """
        Apply oscillation perpendicular to the direction to goal
        """
        # Update oscillation phase
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.oscillation_phase = 2 * math.pi * self.oscillation_frequency * elapsed_time
        
        # Get perpendicular vector to the direction to goal
        perpendicular = np.array([-direction_to_goal[1], direction_to_goal[0]])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # Calculate oscillation offset
        oscillation = perpendicular * self.oscillation_amplitude * math.sin(self.oscillation_phase)
        
        # Apply oscillation to base position
        oscillated_position = base_position + oscillation
        
        return oscillated_position
    
    def calculate_approach_position(self, cow_position, is_engaging):
        """
        Calculate the best position to approach the cow based on engagement status
        """
        # Vector from cow to goal
        cow_to_goal = self.goal_position[:2] - cow_position[:2]
        distance_to_goal = np.linalg.norm(cow_to_goal)
        
        if distance_to_goal > 0.1:
            cow_to_goal_normalized = cow_to_goal / distance_to_goal
        else:
            cow_to_goal_normalized = np.array([1.0, 0.0])  # Default direction if at goal
        
        # Calculate vector directly opposite from cow-to-goal vector
        behind_vector = -cow_to_goal_normalized
        
        # Adjust distance behind based on engagement status
        distance_behind = 2.0 if is_engaging else 3.5
        
        # Calculate base position behind the cow
        base_position_2d = cow_position[:2] + (behind_vector * distance_behind)
        
        # Apply oscillation if engaging, straight path if transiting
        if is_engaging:
            desired_position_2d = self.apply_oscillation(base_position_2d, cow_to_goal_normalized)
        else:
            desired_position_2d = base_position_2d
        
        return desired_position_2d, cow_to_goal_normalized
    
    def timer_callback(self):
        """
        Main logic for herding cattle towards the goal with dynamic altitude
        """
        # Skip if no cattle data received yet
        if not self.cattle_positions:
            self.get_logger().info('No cattle data received yet.')
            return
            
        # Calculate time since last callback
        current_time = time.time()
        dt = current_time - self.last_movement_time
        self.last_movement_time = current_time
        
        # Find furthest cattle from goal
        furthest_cow_id = self.find_furthest_cow()
        
        if furthest_cow_id is None:
            self.get_logger().warn('Could not identify furthest cow!')
            return
            
        furthest_cow_pos = self.cattle_positions[furthest_cow_id]
        
        # Check if we should be engaging or transiting
        is_engaging = self.check_engagement_status(furthest_cow_pos)
        
        # Calculate desired position based on engagement status
        desired_position_2d, cow_to_goal_normalized = self.calculate_approach_position(
            furthest_cow_pos[:2], is_engaging)
        
        # Direction and distance to desired 2D position
        direction_to_desired_2d = desired_position_2d - self.drone_position[:2]
        distance_2d = np.linalg.norm(direction_to_desired_2d)
        
        # Create velocity message
        msg = Twist()
        
        # Calculate velocity vector
        if distance_2d < 0.1:
            # Small random movement to prevent getting stuck
            msg.linear.x = (np.random.random() - 0.5) * 0.2
            msg.linear.y = (np.random.random() - 0.5) * 0.2
        else:
            # Scale velocity based on distance (smoother approach)
            # Use higher speed when in transit mode
            max_speed = self.speed * (1.5 if not is_engaging else 1.0)
            speed_factor = min(1.0, distance_2d / 2.0)
            velocity_scale = max_speed * speed_factor
            
            # Normalize direction vector
            direction_normalized = direction_to_desired_2d / distance_2d
            
            # Set velocity components
            msg.linear.x = direction_normalized[0] * velocity_scale
            msg.linear.y = direction_normalized[1] * velocity_scale
        
        # Apply altitude control with time-based smoothing
        msg = self.altitude_control(msg, dt)
        
        # Publish the velocity command
        self.publisher.publish(msg)
        
        # Log status occasionally
        elapsed_time = time.time() - self.start_time
        if int(elapsed_time * 10) % 50 == 0:  # Log every ~5 seconds
            engagement_status = "Engaging" if is_engaging else "Transiting"
            self.get_logger().info(
                f'Status: {engagement_status} - Altitude: {self.drone_position[2]:.2f}m (Target: {self.current_target_altitude:.1f}m)\n' +
                f'Herding cow {furthest_cow_id} at {furthest_cow_pos[:2]}, Goal at {self.goal_position[:2]}\n' +
                f'Drone at {self.drone_position[:2]}, Distance to desired: {distance_2d:.2f}m'
            )

def main(args=None):
    rclpy.init(args=args)
    wrangler_node = EnhancedWranglerNode()
    
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