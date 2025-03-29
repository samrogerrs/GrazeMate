#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class CattleBoidsNode(Node):
    """
    ROS2 node for cattle movement using Boids algorithm with wrangler avoidance behavior.
    """
    
    def __init__(self):
        super().__init__('cattle_boids_node')
        
        # Parameters
        self.num_cows = 12
        
        # Boids algorithm parameters
        self.separation_weight = 1.5    # Weight for separation force
        self.alignment_weight = 4.0     # Weight for alignment force
        self.cohesion_weight = 0.7      # Weight for cohesion force
        self.wrangler_avoidance_weight = 3.0  # Weight for wrangler avoidance (higher = stronger fleeing)
        self.max_speed = 0.5            # Maximum cow speed
        self.min_speed = 0.1            # Minimum cow speed
        self.perception_radius = 2.0    # How far cows can see other cows
        self.separation_radius = 1.0    # Distance to maintain between cows
        self.wrangler_detection_radius = 2.5  # How far cows can detect wrangler
        self.wrangler_panic_radius = 2.0  # Distance at which cows start to run faster
        self.panic_speed_multiplier = 3.0  # How much faster cows move when panicked
        
        # Cattle state tracking
        self.cattle_positions = {}
        self.cattle_velocities = {}
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
        
        self.get_logger().info('Cattle Boids node initialized')
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions and velocities from odometry
        """
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
        self.cattle_velocities[cattle_id] = np.array([vel.x, vel.y, vel.z])
    
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
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = new_velocity[0]
            cmd.linear.y = new_velocity[1]
            cmd.linear.z = 0.0  # Keep cows on the ground
            
            self.cmd_vel_pubs[cow_id].publish(cmd)
    
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
        """Calculate avoidance force to flee from wrangler"""
        steering = np.zeros(3)
        
        # If wrangler hasn't been detected yet, return zero force
        if not self.wrangler_detected:
            return steering
        
        position = self.cattle_positions[cow_id]
        distance_to_wrangler = np.linalg.norm(position - self.wrangler_position)
        
        # If wrangler is within detection radius, generate avoidance force
        if distance_to_wrangler < self.wrangler_detection_radius and distance_to_wrangler > 0:
            # Vector pointing away from wrangler
            steering = position - self.wrangler_position
            
            # Create a stronger inverse-square response (much stronger at close distances)
            # This gives a more realistic "panic" response
            avoidance_strength = ((self.wrangler_detection_radius / distance_to_wrangler) ** 2) - 1.0
            avoidance_strength = max(0.0, min(10.0, avoidance_strength))  # Clamp between 0 and 10
            
            steering = steering * avoidance_strength
            
            # Don't normalize - allow the length to represent the urgency
            if np.linalg.norm(steering) > 0:
                # Still apply some directional guidance, but keep magnitude
                steering_dir = steering / np.linalg.norm(steering)
                steering = steering_dir * self.max_speed * avoidance_strength
            
            self.get_logger().info(f'Cow {cow_id} avoiding wrangler at distance {distance_to_wrangler:.2f}m, ' 
                                f'force: {np.linalg.norm(steering):.2f}, avoidance: {avoidance_strength:.2f}')
        
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