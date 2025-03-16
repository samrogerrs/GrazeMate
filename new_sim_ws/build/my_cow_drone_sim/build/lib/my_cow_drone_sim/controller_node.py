#!/usr/bin/env python3
import rclpy
import random
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class CowDroneController(Node):
    """
    Controller node for the cow and drone simulation.
    """
    def __init__(self):
        super().__init__('cow_drone_controller')
        self.publisher_ = self.create_publisher(String, 'simulation_status', 10)
        self.drone_vel_publisher = self.create_publisher(
            Twist, '/drone/cmd_vel', 10)
        
        # Create timer for periodic status updates
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Cow Drone Controller has started')
    
    def timer_callback(self):
        """
        Periodic callback to publish status messages.
        """
        msg = String()
        msg.data = 'Simulation running - Cow and Drone'
        self.publisher_.publish(msg)
        self.get_logger().info('Status: %s' % msg.data)
        
        # Example of how to move the drone
        # Uncomment to have the drone move automatically
        self.move_drone()
    
    def move_drone(self):
        """
        Send a velocity command to move the drone.
        """
        vel_msg = Twist()
        # Ensure all velocity components are properly initialized as floats
        vel_msg.linear.x = random.uniform(-1.0, 1.0)
        vel_msg.linear.y = random.uniform(-1.0, 1.0)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 2.0  # Angular velocity (turning)
        
        self.drone_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to drone')

def main(args=None):
    rclpy.init(args=args)
    node = CowDroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()