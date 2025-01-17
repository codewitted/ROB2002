import math
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
        
        # Timer to control movement updates
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        # Store latest laser ranges
        self.laser_ranges = []
        
        # Store our current pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Store the start (x, y) so we can return
        self.start_x = None
        self.start_y = None
        
        # Simple state machine: MOVING or TURNING
        self.state = 'MOVING'
        
        # Used to time how long we turn, etc.
        self.turn_time = 0  
        self.turn_time_target = 0
        
        # Just a basic counter to decide “exploration time”
        self.explore_counter = 0
        
        # For logging or debugging
        self.get_logger().info('MoverBasic node started.')

    def scan_callback(self, msg: LaserScan):
        """Store latest LaserScan data."""
        self.laser_ranges = list(msg.ranges)
    
    def odom_callback(self, msg: Odometry):
        """Extract current position and orientation from odometry."""
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
        
        # If we haven't received any laser data yet, just wait
        if not self.laser_ranges:
            return
        
        # Decide how long we want to explore before heading back
        # (You could pick a more sophisticated completion condition)
        self.explore_counter += 1
        
        # Once we pass some threshold, we’ll try to return home
        # e.g. ~ 2000 cycles at 10 Hz => about 200 seconds
        exploration_done = (self.explore_counter > 2000)
        
        twist = Twist()
        
        if not exploration_done:
            # ---- EXPLORATION/OBSTACLE AVOIDANCE ----
            front_dist = self.get_front_distance()
            
            if self.state == 'MOVING':
                # If something is too close in front, switch to TURNING
                if front_dist < 1.0:  # meters
                    self.state = 'TURNING'
                    # Turn for ~1 second
                    self.turn_time = 0
                    self.turn_time_target = 10  # 10 cycles of 0.1s => 1s
                else:
                    # Move forward
                    twist.linear.x = 0.5  # forward speed
                    twist.angular.z = 0.0
                    
            elif self.state == 'TURNING':
                # Turn in place
                twist.linear.x = 0.0
                twist.angular.z = 0.5  # turn speed
                
                self.turn_time += 1
                if self.turn_time >= self.turn_time_target:
                    self.state = 'MOVING'
        
        else:
            # ---- RETURN TO START ----
            # Naive approach: we simply drive toward the start position
            # by computing a simple heading difference.
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_start < 0.3:
                # Close enough to start => stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info('Reached start again. Stopping.')
            else:
                # Compute desired heading
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                # Normalize angle to [-pi, pi]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                
                # Simple P-control for heading
                angular_speed = 1.0 * yaw_error
                # Limit the angular speed a bit
                if angular_speed > 0.5:
                    angular_speed = 0.5
                if angular_speed < -0.5:
                    angular_speed = -0.5
                
                # Move forward if heading is roughly okay
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                else:
                    twist.linear.x = 0.0
                
                twist.angular.z = angular_speed
        
        # Publish our command
        self.cmd_vel_pub.publish(twist)
    
    def get_front_distance(self):
        """Helper: returns the distance straight ahead from the LaserScan data.
           Assumes the scan array covers 360°, with index = len/2 roughly forward.
        """
        if not self.laser_ranges:
            return float('inf')
        # A typical 360° scan has 0° at the front around middle indices,
        # but this depends on your sensor. Adjust as needed.
        mid_index = len(self.laser_ranges) // 2
        return self.laser_ranges[mid_index]
    

def main(args=None):
    rclpy.init(args=args)
    mover_basic = MoverBasic()
    rclpy.spin(mover_basic)

    mover_basic.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
