from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node

### MANUAL SETTINGS
# r3live_config_file = 'r3live_config_MPIR.yaml'
# r3live_config_file = 'r3live_config_gazebo_sim.yaml'
r3live_config_file = 'r3live_config_substitute.yaml' # MPIR substitute

def generate_launch_description():
  # Subscribed topics
  LiDAR_pointcloud_topic = LaunchConfiguration('LiDAR_pointcloud_topic', default='/laser_cloud_flat')
  IMU_topic = LaunchConfiguration('IMU_topic', default='/livox/imu')
  Image_topic = LaunchConfiguration('Image_topic', default='/camera/image_color')
  map_output_dir = LaunchConfiguration('map_output_dir', default=str(EnvironmentVariable('HOME')) + '/r3live_output')


  r3live_config = os.path.join(
      get_package_share_directory('r3live'),
      'config',
      r3live_config_file
      )

  rviz_config = os.path.join(
      get_package_share_directory('r3live'),
      'config',
      'rviz',
      'r3live_rviz2_config.rviz'
      )

  # Launch r3live_LiDAR_front_end node
  r3live_LiDAR_front_end = Node(
         package='r3live',
         executable='r3live_LiDAR_front_end',
         name='r3live_LiDAR_front_end',
         output='screen',
         parameters=[r3live_config]
      )


  # Launch r3live_mapping node
  r3live_mapping = Node(
         package='r3live',
         executable='r3live_mapping',
         name='r3live_mapping',
         output='screen',
         parameters=[r3live_config]
      )


  # Launch rviz if enabled
  rviz_enabled = LaunchConfiguration('rviz', default='1')
  rviz_visualisation = ExecuteProcess(
    cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config],
    output='log',
    condition=IfCondition(rviz_enabled)
  )

  return LaunchDescription([
    DeclareLaunchArgument('LiDAR_pointcloud_topic', default_value=LiDAR_pointcloud_topic),
    DeclareLaunchArgument('IMU_topic', default_value=IMU_topic),
    DeclareLaunchArgument('Image_topic', default_value=Image_topic),
    DeclareLaunchArgument('map_output_dir', default_value=map_output_dir),
    DeclareLaunchArgument('rviz', default_value=rviz_enabled),

    r3live_LiDAR_front_end,
    r3live_mapping,
    rviz_visualisation
  ])
