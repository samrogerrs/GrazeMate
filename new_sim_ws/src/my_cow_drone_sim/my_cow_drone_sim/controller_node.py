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
        
        # Publishers for two cows
        self.cow1_vel_publisher = self.create_publisher(Twist, '/cow1/cmd_vel', 10)
        self.cow2_vel_publisher = self.create_publisher(Twist, '/cow2/cmd_vel', 10)
        self.cow3_vel_publisher = self.create_publisher(Twist, '/cow3/cmd_vel', 10)
        self.cow4_vel_publisher = self.create_publisher(Twist, '/cow4/cmd_vel', 10)
        self.cow5_vel_publisher = self.create_publisher(Twist, '/cow5/cmd_vel', 10)
        self.cow6_vel_publisher = self.create_publisher(Twist, '/cow6/cmd_vel', 10)
        self.cow7_vel_publisher = self.create_publisher(Twist, '/cow7/cmd_vel', 10)
        self.cow8_vel_publisher = self.create_publisher(Twist, '/cow8/cmd_vel', 10)
        self.cow9_vel_publisher = self.create_publisher(Twist, '/cow9/cmd_vel', 10)
        self.cow10_vel_publisher = self.create_publisher(Twist, '/cow10/cmd_vel', 10)
        self.cow11_vel_publisher = self.create_publisher(Twist, '/cow11/cmd_vel', 10)
        self.cow12_vel_publisher = self.create_publisher(Twist, '/cow12/cmd_vel', 10)
        
        # Create timer for periodic status updates
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Cow Drone Controller has started')

    def timer_callback(self):
        """
        Periodic callback to publish status messages.
        """
        msg = String()
        msg.data = 'Simulation running - Two Cows'
        self.publisher_.publish(msg)
        self.get_logger().info('Status: %s' % msg.data)
        
        # Move both cows independently
        self.move_cow1()
        self.move_cow2()
        self.move_cow3()
        self.move_cow4()
        self.move_cow5()
        self.move_cow6()
        self.move_cow7()
        self.move_cow8()
        self.move_cow9()
        self.move_cow10()
        self.move_cow11()
        self.move_cow12()

    def move_cow1(self):
        """
        Send a velocity command to move cow1.
        """
        vel_msg = Twist()
        # Ensure all velocity components are properly initialized as floats
        vel_msg.linear.x = random.uniform(-1.0, 1.0)
        vel_msg.linear.y = random.uniform(-1.0, 1.0)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 2.0  # Angular velocity (turning)
        self.cow1_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow1')
    
    def move_cow2(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow2_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow2')

    def move_cow3(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow3_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow3')

    def move_cow4(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow4_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow4')

    def move_cow5(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow5_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow5')

    def move_cow6(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow6_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow6')


    def move_cow7(self):
        """
        Send a velocity command to move cow1.
        """
        vel_msg = Twist()
        # Ensure all velocity components are properly initialized as floats
        vel_msg.linear.x = random.uniform(-1.0, 1.0)
        vel_msg.linear.y = random.uniform(-1.0, 1.0)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 2.0  # Angular velocity (turning)
        self.cow7_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow7')
    
    def move_cow8(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow8_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow8')

    def move_cow9(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow9_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow9')

    def move_cow10(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow10_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow10')

    def move_cow11(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow11_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow11')

    def move_cow12(self):
        """
        Send a velocity command to move cow2.
        """
        vel_msg = Twist()
        # Different movement pattern for the second cow
        vel_msg.linear.x = random.uniform(-0.8, 0.8)
        vel_msg.linear.y = random.uniform(-0.8, 0.8)
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = -1.5  # Different angular velocity for variety
        self.cow12_vel_publisher.publish(vel_msg)
        self.get_logger().info('Sent velocity command to cow12')

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