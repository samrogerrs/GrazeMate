#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

class CattleBoidsNode(Node):
    """
    ROS2 node for cattle movement using Boids algorithm with wrangler avoidance behavior.
    Cattle now face the direction they are moving.
    """
    
    def __init__(self):
        super().__init__('cattle_boids_node')
        
        # Parameters
        self.num_cows = 12
        
        # Boids algorithm parameters
        self.separation_weight = 0.7    # Weight for separation force
        self.alignment_weight = 4.0     # Weight for alignment force
        self.cohesion_weight = 0.7      # Weight for cohesion force
        self.wrangler_avoidance_weight = 3.0  # Weight for wrangler avoidance (higher = stronger fleeing)
        self.max_speed = 2.5            # Maximum cow speed
        self.min_speed = 0.6            # Minimum cow speed
        self.perception_radius = 2.0    # How far cows can see other cows
        self.separation_radius = 1.0    # Distance to maintain between cows
        self.wrangler_detection_radius = 4  # How far cows can detect wrangler
        self.wrangler_panic_radius = 3.0  # Distance at which cows start to run faster
        self.panic_speed_multiplier = 3.0  # How much faster cows move when panicked
        
        # Rotation parameters
        self.rotation_speed = 2.0       # Maximum angular velocity (radians/sec)
        
        # Cattle state tracking
        self.cattle_positions = {}
        self.cattle_velocities = {}
        self.cattle_orientations = {}   # Track current orientation of each cow
        self.wrangler_position = np.array([0.0, 0.0, 0.0])  # Default position
        self.wrangler_detected = False
        
        # Create publishers and subscribers
        self.cmd_vel_pubs = {}
        
        # Create publishers for cow velocity commands
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.cmd_vel_pubs[cow_id] = self.create_publisher(
                Twist, 
                f'/{cow_id}/cmd_vel', 
                10
            )
            
            # Subscribe to cow odometry
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
        
        # Subscribe to wrangler position
        self.create_subscription(
            Odometry,
            '/drone/odom',  # Assuming wrangler publishes to this topic
            self.wrangler_callback,
            10
        )
        
        # Create timer for Boids algorithm update
        self.update_timer = self.create_timer(0.1, self.update_boids)
        
        self.get_logger().info('Cattle Boids node initialized with orientation control')
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions, velocities, and orientations from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        
        # Extract current orientation (yaw) from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Calculate yaw (heading) from quaternion
        # Formula: atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw = -math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        self.cattle_velocities[cattle_id] = np.array([vel.x, vel.y, vel.z])
        self.cattle_orientations[cattle_id] = yaw
    
    def wrangler_callback(self, msg):
        """
        Update wrangler position from odometry
        """
        pos = msg.pose.pose.position
        self.wrangler_position = np.array([pos.x, pos.y, pos.z])
        self.wrangler_detected = True
        self.get_logger().debug(f'Wrangler detected at {self.wrangler_position}')
    
    def update_boids(self):
        """
        Apply Boids algorithm to control cattle movement with wrangler avoidance
        Also rotate cattle to face their direction of movement
        """
        # Skip if we don't have enough data yet
        if len(self.cattle_positions) < self.num_cows:
            return
        
        # Calculate new velocities for each cow using Boids algorithm
        for cow_id in self.cattle_positions:
            # Get current position and velocity
            position = self.cattle_positions[cow_id]
            velocity = self.cattle_velocities[cow_id]
            
            # Calculate three Boids forces
            separation = self.calculate_separation(cow_id)
            alignment = self.calculate_alignment(cow_id)
            cohesion = self.calculate_cohesion(cow_id)
            
            # Calculate wrangler avoidance force
            wrangler_avoidance = self.calculate_wrangler_avoidance(cow_id)
            
            # Combine forces with weights
            acceleration = (
                self.separation_weight * separation +
                self.alignment_weight * alignment +
                self.cohesion_weight * cohesion +
                self.wrangler_avoidance_weight * wrangler_avoidance
            )
            
            # Update velocity
            new_velocity = velocity + acceleration
            
            # Determine speed limit based on wrangler proximity
            current_max_speed = self.max_speed
            
            # If wrangler is detected and close, increase speed (panic mode)
            if self.wrangler_detected:
                distance_to_wrangler = np.linalg.norm(position - self.wrangler_position)
                if distance_to_wrangler < self.wrangler_panic_radius:
                    current_max_speed = self.max_speed * self.panic_speed_multiplier
                    self.get_logger().debug(f'Cow {cow_id} is panicking! Speed limit: {current_max_speed}')
            
            # Limit speed
            speed = np.linalg.norm(new_velocity)
            if speed > current_max_speed:
                new_velocity = (new_velocity / speed) * current_max_speed
            elif speed < self.min_speed and speed > 0:
                new_velocity = (new_velocity / speed) * self.min_speed
            
            # Create command
            cmd = Twist()
            cmd.linear.x = new_velocity[0]
            cmd.linear.y = new_velocity[1]
            cmd.linear.z = 0.0  # Keep cows on the ground
            
            # Calculate desired orientation (yaw) based on velocity direction
            # Only calculate new yaw if the cow is actually moving
            if np.linalg.norm(new_velocity[:2]) > 0.1:  # Only if moving faster than threshold
                # Calculate desired yaw angle from velocity vector
                desired_yaw = math.atan2(new_velocity[1], new_velocity[0])
                
                # Get current orientation
                current_yaw = self.cattle_orientations.get(cow_id, 0.0)
                
                # Calculate the shortest angular distance to the desired orientation
                yaw_diff = self.normalize_angle(desired_yaw - current_yaw)
                
                # Set angular velocity to rotate towards the desired orientation
                # Limit the rotation speed
                angular_speed = min(abs(yaw_diff) * 2.0, self.rotation_speed)
                if yaw_diff < 0:
                    angular_speed = -angular_speed
                
                cmd.angular.z = angular_speed
                
                self.get_logger().debug(f'Cow {cow_id} - Current yaw: {current_yaw:.2f}, ' 
                                       f'Desired yaw: {desired_yaw:.2f}, ' 
                                       f'Angular speed: {angular_speed:.2f}')
            else:
                # If not moving, don't rotate
                cmd.angular.z = 0.0
            
            # Publish command
            self.cmd_vel_pubs[cow_id].publish(cmd)
    
    def normalize_angle(self, angle):
        """
        Normalize angle to be between -pi and pi
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_separation(self, cow_id):
        """
        Calculate separation force to avoid crowding neighbors
        """
        steering = np.zeros(3)
        position = self.cattle_positions[cow_id]
        count = 0
        
        for other_id, other_pos in self.cattle_positions.items():
            if other_id != cow_id:
                distance = np.linalg.norm(position - other_pos)
                
                if distance < self.separation_radius and distance > 0:
                    # Vector pointing away from neighbor
                    diff = position - other_pos
                    diff = diff / distance  # Normalize
                    steering += diff
                    count += 1
        
        if count > 0:
            steering /= count
            
            # Set magnitude
            if np.linalg.norm(steering) > 0:
                steering = (steering / np.linalg.norm(steering)) * self.max_speed
                steering -= self.cattle_velocities[cow_id]
        
        return steering
    
    def calculate_alignment(self, cow_id):
        """
        Calculate alignment force to steer towards average heading of neighbors
        """
        steering = np.zeros(3)
        count = 0
        position = self.cattle_positions[cow_id]
        
        for other_id, other_pos in self.cattle_positions.items():
            if other_id != cow_id:
                distance = np.linalg.norm(position - other_pos)
                
                if distance < self.perception_radius:
                    steering += self.cattle_velocities[other_id]
                    count += 1
        
        if count > 0:
            steering /= count
            
            # Set magnitude
            if np.linalg.norm(steering) > 0:
                steering = (steering / np.linalg.norm(steering)) * self.max_speed
                steering -= self.cattle_velocities[cow_id]
        
        return steering
    
    def calculate_cohesion(self, cow_id):
        """
        Calculate cohesion force to move toward center of mass of neighbors
        """
        steering = np.zeros(3)
        count = 0
        position = self.cattle_positions[cow_id]
        
        for other_id, other_pos in self.cattle_positions.items():
            if other_id != cow_id:
                distance = np.linalg.norm(position - other_pos)
                
                if distance < self.perception_radius:
                    steering += other_pos
                    count += 1
        
        if count > 0:
            steering /= count
            
            # Vector pointing to center of mass
            steering -= position
            
            # Set magnitude
            if np.linalg.norm(steering) > 0:
                steering = (steering / np.linalg.norm(steering)) * self.max_speed
                steering -= self.cattle_velocities[cow_id]
        
        return steering
    
    def calculate_wrangler_avoidance(self, cow_id):
        """
        Calculate avoidance force to flee directly away from wrangler
        """
        steering = np.zeros(3)
        
        # If wrangler hasn't been detected yet, return zero force
        if not self.wrangler_detected:
            return steering
        
        position = self.cattle_positions[cow_id]
        distance_to_wrangler = np.linalg.norm(position - self.wrangler_position)
        
        # If wrangler is within detection radius, generate avoidance force
        if distance_to_wrangler < self.wrangler_detection_radius and distance_to_wrangler > 0:
            # Vector pointing directly away from wrangler
            steering = position - self.wrangler_position
            
            # Normalize to get the direction vector
            if np.linalg.norm(steering) > 0:
                steering = steering / np.linalg.norm(steering)
                
                # Scale by max speed directly - no inverse square response
                # This makes cows always run directly away at consistent speed
                flee_speed = self.max_speed
                
                # Apply panic speed multiplier if within panic radius
                if distance_to_wrangler < self.wrangler_panic_radius:
                    flee_speed = self.max_speed * self.panic_speed_multiplier
                
                # Set the magnitude to the flee speed
                steering = steering * flee_speed
                
                self.get_logger().info(f'Cow {cow_id} fleeing directly from wrangler at distance {distance_to_wrangler:.2f}m, ' 
                                    f'speed: {flee_speed:.2f}')
        
        return steering

def main(args=None):
    rclpy.init(args=args)
    node = CattleBoidsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()