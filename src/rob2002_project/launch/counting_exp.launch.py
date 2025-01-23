#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Arguments for environment and method
    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='env1',
        description='Which environment to load: env1, env2, or env3'
    )

    method_arg = DeclareLaunchArgument(
        'method',
        default_value='baseline',
        description='Which navigation approach to run: baseline or enhanced'
    )

    environment = LaunchConfiguration('environment')
    method = LaunchConfiguration('method')

    # Optional: If you are using Gazebo or a simulator, you might
    # include a .world or .launch that sets up environment1, environment2, etc.
    # For example:
    #
    # world_file = PathJoinSubstitution([
    #   get_package_share_directory('rob2002_tutorial'),
    #   'worlds',
    #   environment,  # e.g., 'env1.world'
    # ])
    #
    # environment_include = IncludeLaunchDescription(
    #   PythonLaunchDescriptionSource(os.path.join(
    #       get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'
    #   )),
    #   launch_arguments={'world': world_file}.items(),
    # )

    # The CombinedDetector node
    combined_detector_node = Node(
        package='rob2002_tutorial',
        executable='combined_detector',
        name='combined_detector',
        output='screen'
    )

    # The Counter3D node
    counter_3d_node = Node(
        package='rob2002_tutorial',
        executable='counter_3d',
        name='counter_3d',
        output='screen'
    )

    # The navigation approach can be chosen by the `method` argument
    # e.g., `baseline_sweeper` or `enhanced_sweeper`.
    # We'll assume you have two separate executables: "baseline_sweeper" & "enhanced_sweeper".
    sweeper_node = Node(
        package='rob2002_tutorial',
        executable=method,  # picks whichever is set by method:=baseline or method:=enhanced
        name='sweeper',
        output='screen'
    )

    # Put it all together in a LaunchDescription
    ld = LaunchDescription()

    ld.add_action(environment_arg)
    ld.add_action(method_arg)

    # Optional: if you want to automatically load the correct environment/world:
    # ld.add_action(environment_include)

    ld.add_action(combined_detector_node)
    ld.add_action(counter_3d_node)
    ld.add_action(sweeper_node)

    return ld
