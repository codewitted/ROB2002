#!/usr/bin/env python3

import rclpy
import random
import os
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory

class RandomEnvironmentSpawner(Node):
    def __init__(self):
        super().__init__('random_environment_spawner')
        
        # Create a client for the /spawn_entity service
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        
        # Wait until the service is available
        while not self.spawn_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')
        
        # Paths to your SDF models (adjust to your actual paths)
        # You can put these models in a ROS package, or use an absolute path.
        # Here we assume you have a package "my_gazebo_models" with the SDF inside "models" folder.
        pkg_path = get_package_share_directory('my_gazebo_models')
        self.box_red_sdf_path = os.path.join(pkg_path, 'models', 'box_red', 'model.sdf')
        self.box_green_sdf_path = os.path.join(pkg_path, 'models', 'box_green', 'model.sdf')
        self.box_blue_sdf_path = os.path.join(pkg_path, 'models', 'box_blue', 'model.sdf')
        
        # For robot R
        self.robot_sdf_path = os.path.join(pkg_path, 'models', 'robot_r', 'model.sdf')
        
        # Number of boxes to spawn for each color
        self.num_red = 3
        self.num_green = 3
        self.num_blue = 3
        
        # Bounds within which we place boxes (adjust as needed)
        self.min_x, self.max_x = -4.0, 4.0
        self.min_y, self.max_y = -4.0, 4.0
        
        # Start the spawning routine
        self.spawn_all()
    
    def spawn_all(self):
        """
        Spawns 3 red, 3 green, and 3 blue boxes at random x,y positions,
        plus the robot at a fixed start pose.
        """
        # 1) Spawn the robot at a chosen start
        self.spawn_robot('robot_r', self.robot_sdf_path, x=0.0, y=-4.0, yaw=0.0)
        
        # 2) Spawn red boxes
        for i in range(self.num_red):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            name = f'box_red_{i}'
            self.spawn_box(name, self.box_red_sdf_path, x, y)
        
        # 3) Spawn green boxes
        for i in range(self.num_green):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            name = f'box_green_{i}'
            self.spawn_box(name, self.box_green_sdf_path, x, y)
        
        # 4) Spawn blue boxes
        for i in range(self.num_blue):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            name = f'box_blue_{i}'
            self.spawn_box(name, self.box_blue_sdf_path, x, y)
        
        self.get_logger().info('Done spawning boxes and robot.')
    
    def spawn_robot(self, name, sdf_path, x=0.0, y=0.0, yaw=0.0, z=0.0):
        """
        Spawns the robot at the specified pose.
        """
        with open(sdf_path, 'r') as f:
            sdf_contents = f.read()
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf_contents
        req.robot_namespace = name
        # Set pose
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        # Convert yaw to quaternion or use Euler
        # For simplicity we can just set orientation.z, orientation.w if yaw is small
        import math
        half_yaw = yaw / 2.0
        req.initial_pose.orientation.z = math.sin(half_yaw)
        req.initial_pose.orientation.w = math.cos(half_yaw)
        
        # Call the service (blocking)
        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned robot: {name}')
        else:
            self.get_logger().error(f'Failed to spawn robot: {name}')
    
    def spawn_box(self, name, sdf_path, x, y, z=0.0):
        """
        Spawns a single SDF box model at (x, y).
        """
        with open(sdf_path, 'r') as f:
            sdf_contents = f.read()
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf_contents
        req.robot_namespace = name
        
        # Pose
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        # random orientation about Z if you want
        import math
        yaw = random.uniform(0, math.pi*2)
        half_yaw = yaw / 2.0
        req.initial_pose.orientation.z = math.sin(half_yaw)
        req.initial_pose.orientation.w = math.cos(half_yaw)
        
        # Call the service
        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Spawned {name} at ({x:.2f}, {y:.2f}) yaw={yaw:.2f}')
        else:
            self.get_logger().error(f'Failed to spawn {name} at ({x}, {y})')

def main(args=None):
    rclpy.init(args=args)
    node = RandomEnvironmentSpawner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
