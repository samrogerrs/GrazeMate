#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
import numpy as np
from std_msgs.msg import Bool

class EnvironmentNode(Node):
    """
    ROS2 node for creating and managing the environment with fences for cattle simulation.
    This node:
    1. Creates and publishes fence visualizations
    2. Detects when cattle are approaching fences and publishes boundary forces
    3. Manages fence configuration (rectangular paddock by default)
    """
    
    def __init__(self):
        super().__init__('environment_node')
        
        # Environment parameters
        self.declare_parameter('fence_x_min', -15.0)
        self.declare_parameter('fence_x_max', 15.0)
        self.declare_parameter('fence_y_min', -15.0)
        self.declare_parameter('fence_y_max', 15.0)
        self.declare_parameter('fence_height', 1.5)
        self.declare_parameter('fence_visualization_resolution', 0.5)  # Points per meter
        self.declare_parameter('boundary_force_distance', 3.0)  # Distance at which boundary force starts
        self.declare_parameter('boundary_force_max', 5.0)  # Maximum boundary force
        
        # Get parameters
        self.fence_x_min = self.get_parameter('fence_x_min').value
        self.fence_x_max = self.get_parameter('fence_x_max').value
        self.fence_y_min = self.get_parameter('fence_y_min').value
        self.fence_y_max = self.get_parameter('fence_y_max').value
        self.fence_height = self.get_parameter('fence_height').value
        self.viz_resolution = self.get_parameter('fence_visualization_resolution').value
        self.boundary_force_distance = self.get_parameter('boundary_force_distance').value
        self.boundary_force_max = self.get_parameter('boundary_force_max').value
        
        # Cattle positions tracking
        self.cattle_positions = {}
        self.num_cows = 12  # Should match the value in cattle_boids_node
        
        # Publishers
        self.fence_marker_pub = self.create_publisher(
            MarkerArray,
            '/environment/fences',
            10
        )
        
        # Boundary force publishers - one for each cow
        self.boundary_force_pubs = {}
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.boundary_force_pubs[cow_id] = self.create_publisher(
                Odometry,  # Using Odometry msg to publish forces as linear velocity
                f'/{cow_id}/boundary_force',
                10
            )
        
        # Subscribe to cattle positions
        for i in range(1, self.num_cows + 1):
            cow_id = f'cow{i}'
            self.create_subscription(
                Odometry,
                f'/{cow_id}/odom',
                lambda msg, id=cow_id: self.cattle_callback(msg, id),
                10
            )
        
        # Create timer for visualization updates (lower frequency)
        self.create_timer(1.0, self.publish_fence_markers)
        
        # Create timer for boundary force calculations (higher frequency)
        self.create_timer(0.1, self.calculate_boundary_forces)
        
        self.get_logger().info('Environment node initialized')
        self.get_logger().info(f'Fence boundaries: X: {self.fence_x_min} to {self.fence_x_max}, Y: {self.fence_y_min} to {self.fence_y_max}')
    
    def cattle_callback(self, msg, cattle_id):
        """
        Update cattle positions from odometry
        """
        pos = msg.pose.pose.position
        self.cattle_positions[cattle_id] = np.array([pos.x, pos.y, pos.z])
    
    def publish_fence_markers(self):
        """
        Create and publish visualization markers for the fences
        """
        marker_array = MarkerArray()
        
        # Create four fence sections (one for each side of the rectangular paddock)
        # North fence (y_max)
        north_fence = self.create_fence_marker(
            0,
            [self.fence_x_min, self.fence_y_max, 0],
            [self.fence_x_max, self.fence_y_max, 0],
            'north_fence'
        )
        marker_array.markers.append(north_fence)
        
        # South fence (y_min)
        south_fence = self.create_fence_marker(
            1,
            [self.fence_x_min, self.fence_y_min, 0],
            [self.fence_x_max, self.fence_y_min, 0],
            'south_fence'
        )
        marker_array.markers.append(south_fence)
        
        # East fence (x_max)
        east_fence = self.create_fence_marker(
            2,
            [self.fence_x_max, self.fence_y_min, 0],
            [self.fence_x_max, self.fence_y_max, 0],
            'east_fence'
        )
        marker_array.markers.append(east_fence)
        
        # West fence (x_min)
        west_fence = self.create_fence_marker(
            3,
            [self.fence_x_min, self.fence_y_min, 0],
            [self.fence_x_min, self.fence_y_max, 0],
            'west_fence'
        )
        marker_array.markers.append(west_fence)
        
        # Add fence posts at corners
        post_radius = 0.2
        post_height = self.fence_height + 0.3  # Posts are slightly taller than fence
        
        # Northwest corner post
        nw_post = self.create_post_marker(
            4,
            [self.fence_x_min, self.fence_y_max, 0],
            post_radius,
            post_height,
            'nw_post'
        )
        marker_array.markers.append(nw_post)
        
        # Northeast corner post
        ne_post = self.create_post_marker(
            5,
            [self.fence_x_max, self.fence_y_max, 0],
            post_radius,
            post_height,
            'ne_post'
        )
        marker_array.markers.append(ne_post)
        
        # Southeast corner post
        se_post = self.create_post_marker(
            6,
            [self.fence_x_max, self.fence_y_min, 0],
            post_radius,
            post_height,
            'se_post'
        )
        marker_array.markers.append(se_post)
        
        # Southwest corner post
        sw_post = self.create_post_marker(
            7,
            [self.fence_x_min, self.fence_y_min, 0],
            post_radius,
            post_height,
            'sw_post'
        )
        marker_array.markers.append(sw_post)
        
        # Add fence posts along the fence lines
        post_spacing = 5.0  # Space between posts in meters
        
        # Posts along north fence
        x_positions = np.arange(self.fence_x_min + post_spacing, 
                               self.fence_x_max, 
                               post_spacing)
        marker_id = 8
        for x in x_positions:
            post = self.create_post_marker(
                marker_id,
                [x, self.fence_y_max, 0],
                post_radius,
                post_height,
                f'north_post_{marker_id}'
            )
            marker_array.markers.append(post)
            marker_id += 1
        
        # Posts along south fence
        for x in x_positions:
            post = self.create_post_marker(
                marker_id,
                [x, self.fence_y_min, 0],
                post_radius,
                post_height,
                f'south_post_{marker_id}'
            )
            marker_array.markers.append(post)
            marker_id += 1
        
        # Posts along east fence
        y_positions = np.arange(self.fence_y_min + post_spacing, 
                               self.fence_y_max, 
                               post_spacing)
        for y in y_positions:
            post = self.create_post_marker(
                marker_id,
                [self.fence_x_max, y, 0],
                post_radius,
                post_height,
                f'east_post_{marker_id}'
            )
            marker_array.markers.append(post)
            marker_id += 1
        
        # Posts along west fence
        for y in y_positions:
            post = self.create_post_marker(
                marker_id,
                [self.fence_x_min, y, 0],
                post_radius,
                post_height,
                f'west_post_{marker_id}'
            )
            marker_array.markers.append(post)
            marker_id += 1
        
        # Publish marker array
        self.fence_marker_pub.publish(marker_array)
    
    def create_fence_marker(self, marker_id, start_point, end_point, namespace):
        """
        Create a fence marker as a line strip
        """
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set fence color (wooden fence brown)
        marker.color = ColorRGBA()
        marker.color.r = 0.55
        marker.color.g = 0.27
        marker.color.b = 0.07
        marker.color.a = 1.0
        
        # Set fence scale
        marker.scale.x = 0.1  # Line width
        
        # Generate points along the fence
        start = np.array(start_point)
        end = np.array(end_point)
        fence_length = np.linalg.norm(end - start)
        num_points = int(fence_length * self.viz_resolution) + 2  # +2 for start and end points
        
        # Create multiple horizontal lines to represent fence slats
        num_slats = 3
        for slat in range(num_slats):
            height = slat * (self.fence_height / (num_slats - 1))  # Distribute slats evenly
            
            for i in range(num_points):
                t = i / (num_points - 1)
                point = start * (1 - t) + end * t
                point[2] = height  # Set z coordinate to slat height
                
                p = Point()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = float(point[2])
                marker.points.append(p)
            
            # Add a spacer point (outside view) to separate slats
            p = Point()
            p.x = 1000.0  # Far away point
            p.y = 1000.0
            p.z = 1000.0
            marker.points.append(p)
        
        marker.lifetime.sec = 0  # 0 = forever
        return marker
    
    def create_post_marker(self, marker_id, position, radius, height, namespace):
        """
        Create a fence post marker as a cylinder
        """
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + height / 2  # Center of cylinder
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set post color (darker wooden post)
        marker.color = ColorRGBA()
        marker.color.r = 0.3
        marker.color.g = 0.2
        marker.color.b = 0.1
        marker.color.a = 1.0
        
        # Set post scale
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = height
        
        marker.lifetime.sec = 0  # 0 = forever
        return marker
    
    def calculate_boundary_forces(self):
        """
        Calculate boundary forces for each cow to prevent them from passing through fences
        """
        if not self.cattle_positions:
            return
            
        for cow_id, position in self.cattle_positions.items():
            # Calculate distance to each boundary
            distance_to_north = self.fence_y_max - position[1]
            distance_to_south = position[1] - self.fence_y_min
            distance_to_east = self.fence_x_max - position[0]
            distance_to_west = position[0] - self.fence_x_min
            
            # Initialize boundary force vector
            force = np.zeros(3)
            
            # Add force component for each boundary if cow is close enough
            # North boundary
            if distance_to_north < self.boundary_force_distance:
                force_magnitude = self.calculate_force_magnitude(distance_to_north)
                force[1] -= force_magnitude  # Push south
                self.get_logger().debug(f'Cow {cow_id} near north fence, force: {force_magnitude:.2f}')
            
            # South boundary
            if distance_to_south < self.boundary_force_distance:
                force_magnitude = self.calculate_force_magnitude(distance_to_south)
                force[1] += force_magnitude  # Push north
                self.get_logger().debug(f'Cow {cow_id} near south fence, force: {force_magnitude:.2f}')
            
            # East boundary
            if distance_to_east < self.boundary_force_distance:
                force_magnitude = self.calculate_force_magnitude(distance_to_east)
                force[0] -= force_magnitude  # Push west
                self.get_logger().debug(f'Cow {cow_id} near east fence, force: {force_magnitude:.2f}')
            
            # West boundary
            if distance_to_west < self.boundary_force_distance:
                force_magnitude = self.calculate_force_magnitude(distance_to_west)
                force[0] += force_magnitude  # Push east
                self.get_logger().debug(f'Cow {cow_id} near west fence, force: {force_magnitude:.2f}')
            
            # If there's a non-zero force, publish it
            if np.linalg.norm(force) > 0:
                # Create and publish odometry message with force as linear velocity
                odom_msg = Odometry()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = "odom"
                
                odom_msg.twist.twist.linear.x = force[0]
                odom_msg.twist.twist.linear.y = force[1]
                odom_msg.twist.twist.linear.z = force[2]
                
                self.boundary_force_pubs[cow_id].publish(odom_msg)
    
    def calculate_force_magnitude(self, distance):
        """
        Calculate the magnitude of boundary force based on distance
        Uses exponential increase as distance decreases
        """
        if distance <= 0:
            return self.boundary_force_max  # Maximum force if at or beyond boundary
        
        # Normalized distance factor (0.0 at boundary, 1.0 at boundary_force_distance)
        distance_factor = min(1.0, distance / self.boundary_force_distance)
        
        # Exponential increase as cow gets closer to boundary
        # Force = max_force * (1 - distance_factor)^2
        force = self.boundary_force_max * (1.0 - distance_factor) ** 2
        
        return force

def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()