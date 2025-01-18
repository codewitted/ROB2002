#!/usr/bin/env python3

import math
import random
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy import qos

from geometry_msgs.msg import Twist, Pose, PoseStamped, Point, Quaternion
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge

#------------------------------------------------------------------------------
# Simple 2D bounding-box class for double-counting avoidance (from DetectorBasic)
#------------------------------------------------------------------------------
class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1

    def __and__(self, other):
        """Check if there's an overlap with another rectangle."""
        if (self.x1 > other.x2 or other.x1 > self.x2 or
            self.y1 > other.y2 or other.y1 > self.y2):
            return False
        return True

#------------------------------------------------------------------------------
# Combined Detector Node:
#  - Detects color blobs (red/green/blue),
#  - Converts them to 3D coordinates (Detector3D style),
#  - Avoids double-counting if a blob overlaps with previously seen bounding boxes.
#------------------------------------------------------------------------------
class CombinedDetector(Node):
    def __init__(self):
        super().__init__('combined_detector')

        self.bridge = CvBridge()
        self.real_robot = False  # Set True if using an actual camera with different topics
        self.min_area_size = 100
        self.global_frame = 'odom'  # Could be 'map' if you have SLAM
        self.visualisation = True

        # For double-counting avoidance:
        self.prev_objects = []  # list of Rectangle from the last frame

        # Camera info / models
        self.ccamera_model = None
        self.dcamera_model = None
        self.image_depth_ros = None
        self.color2depth_aspect = None
        
        # Topics vary by robot; adjust if needed
        ccamera_info_topic = '/limo/depth_camera_link/camera_info'
        dcamera_info_topic = '/limo/depth_camera_link/depth/camera_info'
        cimage_topic       = '/limo/depth_camera_link/image_raw'
        dimage_topic       = '/limo/depth_camera_link/depth/image_raw'
        self.camera_frame  = 'depth_link'

        if self.real_robot:
            ccamera_info_topic = '/camera/color/camera_info'
            dcamera_info_topic = '/camera/depth/camera_info'
            cimage_topic       = '/camera/color/image_raw'
            dimage_topic       = '/camera/depth/image_raw'
            self.camera_frame  = 'camera_color_optical_frame'

        # Subscribers
        self.ccamera_info_sub = self.create_subscription(
            CameraInfo, ccamera_info_topic, 
            self.ccamera_info_callback, qos.qos_profile_sensor_data)
        self.dcamera_info_sub = self.create_subscription(
            CameraInfo, dcamera_info_topic,
            self.dcamera_info_callback, qos.qos_profile_sensor_data)
        self.cimage_sub = self.create_subscription(
            Image, cimage_topic, 
            self.image_color_callback, qos.qos_profile_sensor_data)
        self.dimage_sub = self.create_subscription(
            Image, dimage_topic, 
            self.image_depth_callback, qos.qos_profile_sensor_data)

        # Publisher for 3D object location
        self.object_location_pub = self.create_publisher(
            PoseStamped, '/object_location', qos.qos_profile_parameters)

        # TF buffer/listener for camera->odom transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("CombinedDetector started.")

    def ccamera_info_callback(self, data):
        if self.ccamera_model is None:
            from image_geometry import PinholeCameraModel
            self.ccamera_model = PinholeCameraModel()
            self.ccamera_model.fromCameraInfo(data)
            self.color2depth_calc()

    def dcamera_info_callback(self, data):
        if self.dcamera_model is None:
            from image_geometry import PinholeCameraModel
            self.dcamera_model = PinholeCameraModel()
            self.dcamera_model.fromCameraInfo(data)
            self.color2depth_calc()

    def color2depth_calc(self):
        """Compute the ratio needed to align color image coords onto the depth image."""
        if (self.ccamera_model and self.dcamera_model and 
            self.color2depth_aspect is None):
            # Rough approach: compare horizontal FOV per pixel for color vs. depth
            # color: (atan2(width, 2*fx) / width)
            # depth: (atan2(width, 2*fx) / width)
            c_aspect = (math.atan2(self.ccamera_model.width, 2 * self.ccamera_model.fx()) 
                        / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2 * self.dcamera_model.fx()) 
                        / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(f"Computed color2depth_aspect={self.color2depth_aspect:.3f}")

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_color_callback(self, data):
        # Wait until camera models and depth image are available
        if (self.color2depth_aspect is None or 
            self.image_depth_ros is None or 
            self.ccamera_model is None or self.dcamera_model is None):
            return
        
        # Convert ROS images to OpenCV
        color_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        
        # If real robot camera returns mm instead of meters, convert:
        if self.real_robot:
            depth_img /= 1000.0

        # ----------------------------------------------------------------------
        # 1. Detect color blobs (red, green, blue) => combined mask
        # ----------------------------------------------------------------------
        red_mask   = cv2.inRange(color_img,  (0,   0,  80),  (50,  50, 255))
        green_mask = cv2.inRange(color_img,  (0,  80,   0),  (50, 255,  50))
        blue_mask  = cv2.inRange(color_img,  (80,  0,   0), (255,  50,  50))
        all_mask   = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)

        # Find contours for all color blobs
        contours, _ = cv2.findContours(all_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ----------------------------------------------------------------------
        # 2. For each contour above min_area_size => get bounding rect
        #    => skip if it overlaps with previous detection => otherwise 3D transform & publish
        # ----------------------------------------------------------------------
        new_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_size:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rectA = Rectangle(x, y, x+w, y+h)

            # Check overlap with previously seen objects
            is_new = True
            for rectB in self.prev_objects:
                if rectA & rectB:  # overlap => treat as previously counted
                    is_new = False
                    break

            if not is_new:
                # This contour is likely the same object as before
                continue

            # If it's truly new, do 3D geometry on its centroid
            cx = x + w/2
            cy = y + h/2
            # Depth alignment from color -> depth image coords
            # (similar logic to the original image2camera_tf function)
            dx, dy = self.color_to_depth_coords(cx, cy, color_img, depth_img)
            if dx < 0 or dy < 0 or dx >= depth_img.shape[1] or dy >= depth_img.shape[0]:
                # out of bounds; skip
                continue

            depth_value = depth_img[int(dy), int(dx)]
            if depth_value <= 0.0 or math.isinf(depth_value) or math.isnan(depth_value):
                # invalid depth => skip
                continue

            # project pixel to 3D (in camera frame)
            camera_pose = self.pixel_to_3d_pose(cx, cy, depth_value)

            # transform from camera frame to global_frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame, self.camera_frame, rclpy.time.Time())
                global_pose = do_transform_pose(camera_pose, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            # publish PoseStamped for visualization
            stamped = PoseStamped(
                header=Header(frame_id=self.global_frame),
                pose=global_pose
            )
            self.object_location_pub.publish(stamped)

            # For visualization, draw rectangle on color_img
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (255,255,0), 1)
            cv2.circle(color_img, (int(cx), int(cy)), 3, (255,0,255), -1)

            # Add the bounding box to new_bboxes
            new_bboxes.append(rectA)

        # Update prev_objects to everything we saw in this frame
        # (In DetectorBasic, we keep only "detected_objects" as prev; same approach here.)
        self.prev_objects = new_bboxes

        # ----------------------------------------------------------------------
        # 3. Visualization in OpenCV windows
        # ----------------------------------------------------------------------
        if self.visualisation:
            disp_color = color_img.copy()
            disp_depth = depth_img.copy() / 10.0  # scale depth for display

            disp_color = cv2.resize(disp_color, (0,0), fx=0.5, fy=0.5)
            disp_depth = cv2.resize(disp_depth, (0,0), fx=0.5, fy=0.5)

            cv2.imshow("combined_detector color", disp_color)
            cv2.imshow("combined_detector depth", disp_depth)
            cv2.waitKey(1)

    def color_to_depth_coords(self, cx, cy, color_img, depth_img):
        """
        Given a pixel (cx, cy) in the color image, convert to approximate
        (dx, dy) in the depth image. We use self.color2depth_aspect to scale.
        """
        # Center points of each image
        color_cx = color_img.shape[1] / 2.0
        color_cy = color_img.shape[0] / 2.0
        depth_cx = depth_img.shape[1] / 2.0
        depth_cy = depth_img.shape[0] / 2.0

        # Shift from the color center
        shift_x = cx - color_cx
        shift_y = cy - color_cy

        # Scale by color2depth_aspect
        dx = depth_cx + shift_x * self.color2depth_aspect
        dy = depth_cy + shift_y * self.color2depth_aspect

        return dx, dy

    def pixel_to_3d_pose(self, cx, cy, depth_value):
        """
        Convert pixel coords (cx, cy) and a depth_value (m) into a Pose
        in the camera coordinate frame (z forward, x right, y down, typically).
        """
        from image_geometry import PinholeCameraModel
        # Use the color camera model to project pixel to a ray
        # note: projectPixelTo3dRay expects (u, v) = (x, y)
        ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
        # ray is a 3D direction, scale it by depth
        # (the ray has z=1 at the end, so multiply entire vector by depth_value)
        ray *= depth_value / ray[2]

        pose = Pose()
        pose.position = Point(x=ray[0], y=ray[1], z=ray[2])
        pose.orientation = Quaternion(w=1.0)  # no rotation
        return pose


#------------------------------------------------------------------------------
# WarehouseSweeper Node:
#  - Same motion code from before: avoids obstacles, visits corners, returns home,
#    and handles dead-ends by doing a U-turn.
#------------------------------------------------------------------------------
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
        
        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Laser
        self.laser_ranges = []
        
        # Pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.start_x = None
        self.start_y = None
        
        # Example corners for a ~10x10 area (tweak as needed)
        self.corners = [
            (4.5, -4.5),  # corner A
            (4.5,  4.5),  # corner B
            (-4.5, 4.5),  # corner C
            (-4.5, -4.5)  # corner D
        ]
        self.visited_corners = [False] * len(self.corners)
        self.corner_tolerance = 1.0
        
        # State machine
        self.state = STATE_FORWARD
        
        # Timers
        self.brake_time = 0
        self.brake_time_target = 0
        
        self.turn_time = 0
        self.turn_time_target = 0
        
        self.dead_end_time = 0
        self.dead_end_time_target = 0
        
        self.turn_dir = -1  # default turn right

        # Movement parameters
        self.forward_speed = 0.3
        self.obstacle_threshold = 1.5
        self.close_threshold = 0.7
        self.dead_end_threshold = 0.8
        
        self.get_logger().info('WarehouseSweeper node started.')

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_yaw = yaw
        
        if self.start_x is None and self.start_y is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f'Start pose: ({self.start_x:.2f}, {self.start_y:.2f})')

    def timer_callback(self):
        if not self.laser_ranges:
            return
        
        # Check corner visits
        self.check_corners_visited()
        all_corners_visited = all(self.visited_corners)
        
        if all_corners_visited and self.state not in [STATE_RETURN, STATE_DONE, STATE_DEAD_END]:
            self.state = STATE_RETURN
            self.get_logger().info('All corners visited -> returning home...')
        
        twist = Twist()

        # If in FORWARD/BRAKE/TURN, check for dead-end
        if self.state in [STATE_FORWARD, STATE_BRAKE, STATE_TURN]:
            if self.detect_dead_end():
                self.state = STATE_DEAD_END
                self.dead_end_time = 0
                self.dead_end_time_target = 20  # ~2 sec at 10 Hz
                self.get_logger().info('Dead end detected! U-turn then return home...')
        
        if self.state == STATE_FORWARD:
            front_dist = self.get_front_distance()
            if front_dist < self.obstacle_threshold:
                self.state = STATE_BRAKE
                self.brake_time = 0
                self.brake_time_target = 5  # 0.5s
            else:
                twist.linear.x = self.forward_speed
        
        elif self.state == STATE_BRAKE:
            # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.brake_time += 1
            if self.brake_time >= self.brake_time_target:
                self.decide_turn_direction()
                self.state = STATE_TURN
                self.turn_time = 0
                self.turn_time_target = 10  # 1s

        elif self.state == STATE_TURN:
            twist.linear.x = 0.0
            twist.angular.z = 0.5 * self.turn_dir
            self.turn_time += 1
            if self.turn_time >= self.turn_time_target:
                self.state = STATE_FORWARD

        elif self.state == STATE_DEAD_END:
            # Slow 180 turn
            twist.linear.x = 0.0
            twist.angular.z = 0.3
            self.dead_end_time += 1
            if self.dead_end_time >= self.dead_end_time_target:
                # then go home
                self.state = STATE_RETURN

        elif self.state == STATE_RETURN:
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            dist_to_start = math.sqrt(dx*dx + dy*dy)
            if dist_to_start < 0.5:
                if all_corners_visited:
                    self.get_logger().info('Back at start, all corners done => DONE!')
                    self.state = STATE_DONE
                else:
                    self.get_logger().info('Back at start, corners remain => FORWARD!')
                    self.state = STATE_FORWARD
            else:
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                
                angular_speed = 1.0 * yaw_error
                angular_speed = max(-0.5, min(0.5, angular_speed))
                
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                else:
                    twist.linear.x = 0.0
                twist.angular.z = angular_speed

        elif self.state == STATE_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def decide_turn_direction(self):
        # Typically turn right unless it's blocked
        right_dist = self.get_right_distance()
        left_dist  = self.get_left_distance()
        
        chosen = -1  # right
        if right_dist < 0.8 or right_dist < left_dist:
            chosen = 1  # left
        
        self.turn_dir = chosen
        side = 'RIGHT' if chosen == -1 else 'LEFT'
        self.get_logger().info(f'Braking done. Turn {side}.')

    def detect_dead_end(self):
        front = self.get_front_distance()
        left  = self.get_left_distance()
        right = self.get_right_distance()
        if (front < self.dead_end_threshold 
            and left < self.dead_end_threshold
            and right < self.dead_end_threshold):
            return True
        return False

    def get_front_distance(self):
        if not self.laser_ranges:
            return float('inf')
        mid = len(self.laser_ranges)//2
        return self.laser_ranges[mid]

    def get_left_distance(self):
        if not self.laser_ranges:
            return float('inf')
        mid = len(self.laser_ranges)//2
        idx = mid + 90
        if idx >= len(self.laser_ranges):
            idx = len(self.laser_ranges)-1
        return self.laser_ranges[idx]

    def get_right_distance(self):
        if not self.laser_ranges:
            return float('inf')
        mid = len(self.laser_ranges)//2
        idx = mid - 90
        if idx < 0:
            idx = 0
        return self.laser_ranges[idx]

    def check_corners_visited(self):
        for i, (cx, cy) in enumerate(self.corners):
            if not self.visited_corners[i]:
                dist = math.sqrt((cx - self.current_x)**2 + (cy - self.current_y)**2)
                if dist < self.corner_tolerance:
                    self.visited_corners[i] = True
                    self.get_logger().info(f'Visited corner {i+1} at ({cx}, {cy}).')

#------------------------------------------------------------------------------
# Main: Runs both nodes in one Python process with a MultiThreadedExecutor
#------------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    
    sweeper = WarehouseSweeper()
    detector = CombinedDetector()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(sweeper)
    executor.add_node(detector)

    try:
        executor.spin()
    finally:
        sweeper.destroy_node()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
