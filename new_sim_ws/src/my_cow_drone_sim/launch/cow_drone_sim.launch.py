from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('my_cow_drone_sim')
    world_file = os.path.join(pkg_share, 'worlds', 'cow_drone_world.sdf')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Create bridge arguments for all cows
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'
    ]
    
    # Add cmd_vel bridges
    for i in range(1, 13):
        bridge_args.append(f'/cow{i}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist')
    
    # Add odometry bridges
    for i in range(1, 13):
        bridge_args.append(f'/cow{i}/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'),
            
        # Launch Gazebo
        ExecuteProcess(
            cmd=[
                FindExecutable(name='ign'), 'gazebo', '-r', world_file
            ],
            output='screen'),
            
        # Bridge for cow topics
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            arguments=bridge_args,
            output='screen'
        ),
        
        # Launch controller node
        Node(
            package='my_cow_drone_sim',
            executable='controller_node',
            name='cow_drone_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),
    ])