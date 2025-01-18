import math
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

# Possible states
STATE_FORWARD = 'FORWARD'
STATE_BRAKE   = 'BRAKE'
STATE_TURN    = 'TURN'
STATE_RETURN  = 'RETURN_HOME'
STATE_DONE    = 'DONE'

class WarehouseSweeper(Node):
    def __init__(self):
        super().__init__('warehouse_sweeper')
        
        # Publishers & Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        # Timer (10 Hz = 0.1s)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Laser ranges
        self.laser_ranges = []
        
        # Current pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Store starting position (once odom arrives)
        self.start_x = None
        self.start_y = None
        
        # List of corners for a rectangular warehouse
        # Example coordinates for a ~10x10 area:
        self.corners = [
            (4.5, -4.5),  # corner A (bottom-right)
            (4.5,  4.5),  # corner B (top-right)
            (-4.5, 4.5),  # corner C (top-left)
            (-4.5, -4.5)  # corner D (bottom-left)
        ]
        self.visited_corners = [False] * len(self.corners)
        self.corner_tolerance = 1.0  # how close we need to be
        
        # State machine
        self.state = STATE_FORWARD
        
        # Brake/Turn timers
        self.brake_time = 0
        self.brake_time_target = 0
        
        self.turn_time = 0
        self.turn_time_target = 0
        
        # Turning direction (+1 = turn left, -1 = turn right)
        self.turn_dir = -1  # default turn right for “outer wall”
        
        # Movement parameters
        self.forward_speed = 0.3
        self.obstacle_threshold = 1.5  # start to brake if within 1.5m
        self.close_threshold = 0.7     # if within 0.7m, definitely turn
        
        self.get_logger().info('WarehouseSweeper node started.')

    def scan_callback(self, msg: LaserScan):
        """Callback to store latest LaserScan data."""
        self.laser_ranges = list(msg.ranges)
    
    def odom_callback(self, msg: Odometry):
        """Callback to extract robot’s position/orientation from odometry."""
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
        
        # Record start position once
        if self.start_x is None and self.start_y is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f'Starting pose: ({self.start_x:.2f}, {self.start_y:.2f})')

    def timer_callback(self):
        """Main loop at 10 Hz. Handles states & publishes cmd_vel."""
        if not self.laser_ranges:
            # If no laser data, do nothing
            return
        
        # Check if we have visited all corners
        self.check_corners_visited()
        all_corners_visited = all(self.visited_corners)
        
        # If all corners are visited and not yet returning -> switch
        if all_corners_visited and self.state not in [STATE_RETURN, STATE_DONE]:
            self.state = STATE_RETURN
            self.get_logger().info('All corners visited! Returning to start...')
        
        # Build Twist command
        twist = Twist()
        
        if self.state == STATE_FORWARD:
            # Move forward unless obstacle is too close
            front_dist = self.get_front_distance()
            
            # If front obstacle is within threshold => brake
            if front_dist < self.obstacle_threshold:
                self.state = STATE_BRAKE
                self.brake_time = 0
                # brake for ~0.5s => 5 cycles at 10 Hz
                self.brake_time_target = 5
            else:
                twist.linear.x = self.forward_speed
        
        elif self.state == STATE_BRAKE:
            # Stop for short time to “brake”
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.brake_time += 1
            
            if self.brake_time >= self.brake_time_target:
                # Decide turn direction
                self.decide_turn_direction()
                
                # Switch to turning
                self.state = STATE_TURN
                self.turn_time = 0
                # Turn for ~1 sec => 10 cycles at 10 Hz
                self.turn_time_target = 10
        
        elif self.state == STATE_TURN:
            # Execute turning
            twist.linear.x = 0.0
            twist.angular.z = 0.5 * self.turn_dir
            
            self.turn_time += 1
            if self.turn_time >= self.turn_time_target:
                # Done turning => move forward
                self.state = STATE_FORWARD
        
        elif self.state == STATE_RETURN:
            # Attempt to go back to start
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            # If close enough, we are done
            if distance_to_start < 0.5:
                self.get_logger().info('Arrived back at start. Done!')
                self.state = STATE_DONE
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                # Turn to face start
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                # Normalize angle to [-pi, pi]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                
                # Simple P-control for heading
                angular_speed = 1.0 * yaw_error
                # Limit turn rate
                angular_speed = max(-0.5, min(0.5, angular_speed))
                
                # Move forward if roughly facing the start
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                else:
                    twist.linear.x = 0.0
                twist.angular.z = angular_speed
        
        elif self.state == STATE_DONE:
            # No movement
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        # Publish the velocity
        self.cmd_vel_pub.publish(twist)

    def get_front_distance(self):
        """Returns approximate distance straight ahead from LaserScan data."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        return self.laser_ranges[mid_index]

    def get_left_distance(self):
        """Distance ~90° to the left from LaserScan (approx)."""
        if not self.laser_ranges:
            return float('inf')
        # If 0° is front at mid_index, left ~ mid_index + 90 deg worth of bins
        # Exact index depends on your LaserScan's angle_increment. 
        # This is approximate for a 360° scan w/ 720 points => ~2x per degree
        mid_index = len(self.laser_ranges) // 2
        left_index = mid_index + 90  # ~90 deg
        if left_index >= len(self.laser_ranges):
            left_index = len(self.laser_ranges) - 1
        return self.laser_ranges[left_index]

    def get_right_distance(self):
        """Distance ~90° to the right from LaserScan (approx)."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        right_index = mid_index - 90  # ~90 deg
        if right_index < 0:
            right_index = 0
        return self.laser_ranges[right_index]

    def decide_turn_direction(self):
        """
        Decide whether to turn right or left.
        - By default, turn right if we suspect it's the outer boundary.
        - If right side is blocked, turn left.
        """
        front_dist = self.get_front_distance()
        left_dist = self.get_left_distance()
        right_dist = self.get_right_distance()
        
        # If it's truly the outer "grey wall," we generally want to turn right.
        # But if the right side is blocked by an obstacle, turn left.
        # If front < close_threshold => definitely we have to turn, 
        #   but direction might vary if right is also blocked.
        
        # default: turn right
        chosen = -1
        
        # if right side is *very* close or smaller than left side, turn left
        if right_dist < 0.8 or right_dist < left_dist:
            chosen = 1  # turn left

        self.turn_dir = chosen

    def check_corners_visited(self):
        """
        Check each corner to see if we've come within `corner_tolerance`.
        Mark them visited if so.
        """
        for i, (cx, cy) in enumerate(self.corners):
            if not self.visited_corners[i]:
                dist = math.sqrt((cx - self.current_x)**2 + (cy - self.current_y)**2)
                if dist < self.corner_tolerance:
                    self.visited_corners[i] = True
                    self.get_logger().info(f'Visited corner {i+1} at ({cx}, {cy}).')


def main(args=None):
    rclpy.init(args=args)
    node = WarehouseSweeper()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
