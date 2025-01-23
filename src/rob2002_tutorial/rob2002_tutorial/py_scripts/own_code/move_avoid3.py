import math
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

# Possible states
STATE_FORWARD  = 'FORWARD'
STATE_BRAKE    = 'BRAKE'
STATE_TURN     = 'TURN'
STATE_DEAD_END = 'DEAD_END'
STATE_RETURN   = 'RETURN_HOME'
STATE_DONE     = 'DONE'


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
        
        # Example corners for a ~10x10 warehouse
        self.corners = [
            (4.5, -4.5),  # corner A (bottom-right)
            (4.5,  4.5),  # corner B (top-right)
            (-4.5, 4.5),  # corner C (top-left)
            (-4.5, -4.5)  # corner D (bottom-left)
        ]
        self.visited_corners = [False] * len(self.corners)
        self.corner_tolerance = 1.0  # how close we need to be to mark "visited"
        
        # State machine
        self.state = STATE_FORWARD
        
        # Timers for BRAKE / TURN / DEAD_END
        self.brake_time = 0
        self.brake_time_target = 0
        
        self.turn_time = 0
        self.turn_time_target = 0
        
        self.dead_end_time = 0
        self.dead_end_time_target = 0
        
        # Turning direction (+1 = left, -1 = right)
        self.turn_dir = -1  # default to turning right
        
        # Movement parameters
        self.forward_speed = 0.3
        self.obstacle_threshold = 1.5  # start braking if < 1.5m ahead
        self.close_threshold = 0.7     # definitely turn if < 0.7m
        self.dead_end_threshold = 0.8  # if front, left, right < 0.8 => dead-end
        
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
        
        # Check if we've visited all corners
        self.check_corners_visited()
        all_corners_visited = all(self.visited_corners)
        
        # If corners are all visited and not returning, switch
        if all_corners_visited and self.state not in [STATE_RETURN, STATE_DONE, STATE_DEAD_END]:
            self.state = STATE_RETURN
            self.get_logger().info('All corners visited! Returning to start...')
        
        twist = Twist()
        
        # 1. Check for dead-end condition (unless we're already returning or done)
        if self.state in [STATE_FORWARD, STATE_BRAKE, STATE_TURN]:
            if self.detect_dead_end():
                # If we see a dead end, forcibly switch to dead-end routine
                self.state = STATE_DEAD_END
                self.dead_end_time = 0
                # We'll do a slow 180 turn for ~2 seconds => 20 cycles
                self.dead_end_time_target = 20
                self.get_logger().info('Dead end detected! Performing U-turn and returning home...')
        
        # 2. State machine
        if self.state == STATE_FORWARD:
            # Move forward unless obstacle is too close
            front_dist = self.get_front_distance()
            
            if front_dist < self.obstacle_threshold:
                # Switch to BRAKE
                self.state = STATE_BRAKE
                self.brake_time = 0
                self.brake_time_target = 5  # brake ~0.5s
            else:
                twist.linear.x = self.forward_speed
        
        elif self.state == STATE_BRAKE:
            # Stop briefly
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.brake_time += 1
            
            if self.brake_time >= self.brake_time_target:
                self.decide_turn_direction()
                
                self.state = STATE_TURN
                self.turn_time = 0
                self.turn_time_target = 10  # turn for ~1s
        
        elif self.state == STATE_TURN:
            # Turn in place
            twist.linear.x = 0.0
            twist.angular.z = 0.5 * self.turn_dir
            
            self.turn_time += 1
            if self.turn_time >= self.turn_time_target:
                self.state = STATE_FORWARD
        
        elif self.state == STATE_DEAD_END:
            # We do a slow 180 turn in place
            twist.linear.x = 0.0
            twist.angular.z = 0.3  # slower turn to avoid hitting walls
            
            self.dead_end_time += 1
            if self.dead_end_time >= self.dead_end_time_target:
                # After turning around, go back to start
                self.state = STATE_RETURN
        
        elif self.state == STATE_RETURN:
            # Head back to start
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_start < 0.5:
                # Reached start => decide next action
                if all_corners_visited:
                    self.get_logger().info('Arrived at start. All corners visited. DONE!')
                    self.state = STATE_DONE
                else:
                    self.get_logger().info('Arrived at start. Resuming corner visits...')
                    # If corners remain, go forward again
                    self.state = STATE_FORWARD
            else:
                # Turn to face start
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # normalize
                
                # Simple P-control
                angular_speed = 1.0 * yaw_error
                angular_speed = max(-0.5, min(0.5, angular_speed))
                
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                else:
                    twist.linear.x = 0.0
                twist.angular.z = angular_speed
        
        elif self.state == STATE_DONE:
            # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        # Publish
        self.cmd_vel_pub.publish(twist)

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def detect_dead_end(self):
        """Detect a dead end if front, left, and right distances are all < dead_end_threshold."""
        front = self.get_front_distance()
        left = self.get_left_distance()
        right = self.get_right_distance()
        
        # If all are under some threshold, assume it's a dead end
        if front < self.dead_end_threshold and left < self.dead_end_threshold and right < self.dead_end_threshold:
            return True
        return False

    def check_corners_visited(self):
        """Check if we're within 'corner_tolerance' of each corner; mark visited."""
        for i, (cx, cy) in enumerate(self.corners):
            if not self.visited_corners[i]:
                dist = math.sqrt((cx - self.current_x)**2 + (cy - self.current_y)**2)
                if dist < self.corner_tolerance:
                    self.visited_corners[i] = True
                    self.get_logger().info(f'Visited corner {i+1} at ({cx:.1f}, {cy:.1f}).')
    
    def decide_turn_direction(self):
        """
        Decide whether to turn right or left, typically turning right to follow
        the outer wall, unless the right side is also blocked.
        """
        front_dist = self.get_front_distance()
        left_dist  = self.get_left_distance()
        right_dist = self.get_right_distance()
        
        # Default: turn right
        chosen = -1
        
        # If right side is more blocked than left, pick left
        if right_dist < 0.8 or right_dist < left_dist:
            chosen = 1
        
        self.turn_dir = chosen
        if chosen == -1:
            self.get_logger().info('Turning RIGHT.')
        else:
            self.get_logger().info('Turning LEFT.')

    def get_front_distance(self):
        """Returns approximate distance straight ahead from LaserScan."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        return self.laser_ranges[mid_index]
    
    def get_left_distance(self):
        """Approx. distance 90° to left from LaserScan."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        left_index = mid_index + 90
        if left_index >= len(self.laser_ranges):
            left_index = len(self.laser_ranges) - 1
        return self.laser_ranges[left_index]

    def get_right_distance(self):
        """Approx. distance 90° to right from LaserScan."""
        if not self.laser_ranges:
            return float('inf')
        mid_index = len(self.laser_ranges) // 2
        right_index = mid_index - 90
        if right_index < 0:
            right_index = 0
        return self.laser_ranges[right_index]


def main(args=None):
    rclpy.init(args=args)
    node = WarehouseSweeper()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
