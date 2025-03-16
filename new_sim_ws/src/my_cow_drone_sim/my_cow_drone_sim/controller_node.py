#!/usr/bin/env python3
import rclpy
import random
import math
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class CowBoidsControllerSimple(Node):
    """
    Controller node for cattle using boids flocking algorithm.
    This version uses estimated positions instead of relying on odometry feedback.
    """
    def __init__(self):
        super().__init__('cow_boids_controller')
        self.publisher_ = self.create_publisher(String, 'simulation_status', 10)
        
        # Parameters for boids algorithm
        self.cohesion_factor = 0.05      # Attraction to center of mass
        self.separation_factor = 0.05     # Repulsion from other cows
        self.alignment_factor = 0.02      # Velocity matching
        self.separation_distance = 2.0    # Minimum distance between cows
        self.perception_radius = 10.0     # How far cows can "see"
        self.max_speed = 0.8              # Maximum linear velocity
        self.max_angular_speed = 1.5      # Maximum angular velocity
        self.random_factor = 0.1          # Random movement factor
        
        # Store estimated positions of all cows
        self.num_cows = 12
        self.cow_vel_publishers = {}
        self.estimated_positions = {}  # We'll estimate positions based on commanded velocities
        self.estimated_velocities = {}
        
        # Initialize positions from world file (approximately)
        initial_positions = {
            'cow1': np.array([0.0, 2.0, 0.5]),
            'cow2': np.array([5.0, -2.0, 0.5]),
            'cow3': np.array([5.0, -2.0, 0.5]),
            'cow4': np.array([3.0, -3.0, 0.5]),
            'cow5': np.array([6.0, -1.0, 0.5]),
            'cow6': np.array([6.0, 2.0, 0.5]),
            'cow7': np.array([0.0, -2.0, 0.5]),
            'cow8': np.array([-5.0, -2.0, 0.5]),
            'cow9': np.array([-5.0, -2.0, 0.5]),
            'cow10': np.array([-3.0, -3.0, 0.5]),
            'cow11': np.array([-6.0, -1.0, 0.5]),
            'cow12': np.array([-6.0, 2.0, 0.5])
        }
        
        # Create publishers and initialize positions
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.cow_vel_publishers[cow_id] = self.create_publisher(
                Twist, f'/{cow_id}/cmd_vel', 10)
            
            self.estimated_positions[cow_id] = initial_positions[cow_id]
            self.estimated_velocities[cow_id] = np.zeros(3)
        
        # Create timer for periodic updates
        self.update_interval = 0.1  # seconds
        self.timer = self.create_timer(self.update_interval, self.boids_update)
        self.get_logger().info('Cow Boids Controller has started')

    def boids_update(self):
        """
        Update cow movement based on boids algorithm.
        """
        # Publish status message
        msg = String()
        msg.data = 'Boids simulation running with 12 cows'
        self.publisher_.publish(msg)
        
        # Calculate new velocities for each cow using boids algorithm
        for cow_id in self.estimated_positions:
            # Calculate steering forces
            separation = self.calculate_separation(cow_id)
            alignment = self.calculate_alignment(cow_id)
            cohesion = self.calculate_cohesion(cow_id)
            
            # Apply steering forces to velocity
            new_velocity = self.estimated_velocities[cow_id] + separation + alignment + cohesion
            
            # Add some randomness
            new_velocity += np.array([
                random.uniform(-1.0, 1.0) * self.random_factor,
                random.uniform(-1.0, 1.0) * self.random_factor,
                0.0
            ])
            
            # Limit speed
            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = (new_velocity / speed) * self.max_speed
                
            # Apply the new velocity
            self.apply_velocity(cow_id, new_velocity)
            
            # Update estimated position based on velocity command
            self.estimated_velocities[cow_id] = new_velocity
            self.estimated_positions[cow_id] += new_velocity * self.update_interval

    def calculate_separation(self, cow_id):
        """
        Calculate separation force to avoid crowding.
        """
        separation_force = np.zeros(3)
        neighbors = 0
        
        for other_id, position in self.estimated_positions.items():
            if other_id != cow_id:
                distance_vector = self.estimated_positions[cow_id] - position
                distance = np.linalg.norm(distance_vector)
                
                if 0 < distance < self.separation_distance:
                    # Normalize and weight by distance
                    repulsion = distance_vector / distance
                    repulsion = repulsion / distance  # Closer cows have stronger effect
                    separation_force += repulsion
                    neighbors += 1
        
        if neighbors > 0:
            separation_force /= neighbors
            separation_force *= self.separation_factor
            
        return separation_force

    def calculate_alignment(self, cow_id):
        """
        Calculate alignment force to match velocity with neighbors.
        """
        alignment_force = np.zeros(3)
        neighbors = 0
        
        for other_id, velocity in self.estimated_velocities.items():
            if other_id != cow_id:
                distance = np.linalg.norm(
                    self.estimated_positions[cow_id] - self.estimated_positions[other_id])
                
                if distance < self.perception_radius:
                    alignment_force += velocity
                    neighbors += 1
        
        if neighbors > 0:
            alignment_force /= neighbors
            alignment_force -= self.estimated_velocities[cow_id]
            alignment_force *= self.alignment_factor
            
        return alignment_force

    def calculate_cohesion(self, cow_id):
        """
        Calculate cohesion force to move toward center of mass.
        """
        center_of_mass = np.zeros(3)
        neighbors = 0
        
        for other_id, position in self.estimated_positions.items():
            if other_id != cow_id:
                distance = np.linalg.norm(
                    self.estimated_positions[cow_id] - position)
                
                if distance < self.perception_radius:
                    center_of_mass += position
                    neighbors += 1
        
        cohesion_force = np.zeros(3)
        if neighbors > 0:
            center_of_mass /= neighbors
            cohesion_force = center_of_mass - self.estimated_positions[cow_id]
            cohesion_force *= self.cohesion_factor
            
        return cohesion_force

    def apply_velocity(self, cow_id, velocity):
        """
        Send velocity command to the cow.
        """
        vel_msg = Twist()
        vel_msg.linear.x = float(velocity[0])
        vel_msg.linear.y = float(velocity[1])
        vel_msg.linear.z = 0.0
        
        # Calculate angular velocity (Z-axis rotation)
        if abs(velocity[0]) > 0.01 or abs(velocity[1]) > 0.01:
            # Calculate desired heading
            desired_heading = math.atan2(velocity[1], velocity[0])
            
            # For simplicity, we're using a constant angular velocity
            vel_msg.angular.z = self.max_angular_speed if random.random() > 0.5 else -self.max_angular_speed
            vel_msg.angular.z *= 0.5
        else:
            vel_msg.angular.z = 0.0
            
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        
        # Publish velocity command
        self.cow_vel_publishers[cow_id].publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CowBoidsControllerSimple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()