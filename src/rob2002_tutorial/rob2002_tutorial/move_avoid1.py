import math
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


class MoverBasic(Node):
    def __init__(self):
        super().__init__('mover_basic')
        
        # Publisher to cmd_vel (movement commands)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscriber to LaserScan for obstacle avoidance
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)
        
        # Subscriber to Odometry so we can track position and return to start
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        # Timer to control movement updates (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Store the latest laser ranges
        self.laser_ranges = []
        
        # Current pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Remember where we started (to return later)
        self.start_x = None
        self.start_y = None
        
        # Simple state machine for wandering
        self.state = 'MOVING'
        
        # Turning state variables
        self.turn_time = 0
        self.turn_time_target = 0
        self.turn_dir = 1  # +1 = left, -1 = right
        
        # Counter to decide when to finish exploring
        self.explore_counter = 0
        
        self.get_logger().info('MoverBasic node started.')

    def scan_callback(self, msg: LaserScan):
        """Store the latest LaserScan data."""
        self.laser_ranges = list(msg.ranges)
    
    def odom_callback(self, msg: Odometry):
        """Extract current position/orientation from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to roll/pitch/yaw
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_yaw = yaw
        
        # Save start position once
        if self.start_x is None and self.start_y is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f'Starting pose: ({self.start_x:.2f}, {self.start_y:.2f})')

    def timer_callback(self):
        """Called at 10 Hz. Decide how to move based on laser and state."""
        
        if not self.laser_ranges:
            # If no laser data yet, do nothing
            return
        
        # Let the robot explore for ~2000 cycles (~200 seconds at 10Hz)
        self.explore_counter += 1
        exploration_done = (self.explore_counter > 2000)
        
        twist = Twist()
        
        if not exploration_done:
            # ------------------ EXPLORATION/OBSTACLE AVOIDANCE ------------------
            front_dist = self.get_front_distance()
            
            if self.state == 'MOVING':
                # Gradually slow down if we see something within 1.5m
                safe_dist = 1.5
                if front_dist < safe_dist:
                    # Closer than safe_dist => slow down proportionally
                    ratio = front_dist / safe_dist  # 0..1
                    twist.linear.x = max(0.1, 0.3 * ratio)  # never drop below 0.1
                    # If extremely close, switch to turning
                    if front_dist < 0.7:
                        self.switch_to_turning()
                else:
                    # Move forward at normal exploration speed
                    twist.linear.x = 0.3
                
            elif self.state == 'TURNING':
                twist.linear.x = 0.0
                # Turn in place, direction decided when we entered turning
                twist.angular.z = 0.5 * self.turn_dir
                
                self.turn_time += 1
                if self.turn_time >= self.turn_time_target:
                    self.state = 'MOVING'
                    
        else:
            # ------------------ RETURN TO START ------------------
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_start < 0.3:
                # Close enough to starting point => stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info('Reached start again. Stopping.')
            else:
                # Turn towards start
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # normalize
                
                # Simple P-control for heading
                angular_speed = 1.0 * yaw_error
                angular_speed = max(-0.5, min(0.5, angular_speed))  # clamp
                
                # If we are roughly facing the start, move forward
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                else:
                    twist.linear.x = 0.0
                
                twist.angular.z = angular_speed
        
        # Publish our command
        self.cmd_vel_pub.publish(twist)

    def get_front_distance(self):
        """Helper: returns approximate distance straight ahead from LaserScan data."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        return self.laser_ranges[mid_index]
    
    def switch_to_turning(self):
        """Switch state to TURNING and pick a random turn direction."""
        self.state = 'TURNING'
        self.turn_time = 0
        # Turn in place for ~1 second at 10 Hz => 10 cycles
        self.turn_time_target = 10
        
        # Randomly pick left or right
        self.turn_dir = 1 if random.random() > 0.5 else -1
        self.get_logger().info(f'Switching to TURNING. turn_dir={self.turn_dir}')


def main(args=None):
    rclpy.init(args=args)
    mover_basic = MoverBasic()
    rclpy.spin(mover_basic)

    mover_basic.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
