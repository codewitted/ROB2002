#!/usr/bin/env python3

import math
import numpy as np
import csv
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy import qos

from geometry_msgs.msg import Twist, Pose, PoseStamped, Point, Quaternion, PoseArray
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge

#------------------------------------------------------------------------------
# Simple 2D bounding-box class for double-detection avoidance
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

    def overlaps(self, other) -> bool:
        """Check if there's an overlap with another rectangle (2D)."""
        if (self.x1 > other.x2 or other.x1 > self.x2 or
            self.y1 > other.y2 or other.y1 > self.y2):
            return False
        return True

#------------------------------------------------------------------------------
# Detector Node
#   - Detects color blobs in the RGB image (red/green/blue).
#   - Avoids "double detection" via bounding-box overlap in 2D image space.
#   - Converts new blobs' centroids to 3D using depth image + camera model.
#   - Publishes PoseStamped in a global frame (e.g., 'odom' or 'map').
#
#   Topics (default):
#     color camera: /limo/depth_camera_link/image_raw
#     depth camera: /limo/depth_camera_link/depth/image_raw
#     color cam info: /limo/depth_camera_link/camera_info
#     depth cam info: /limo/depth_camera_link/depth/camera_info
#------------------------------------------------------------------------------
class CombinedDetector(Node):
    def __init__(self):
        super().__init__('combined_detector')

        self.bridge = CvBridge()
        self.real_robot = False  # If True, adjust for real camera scale (mm->m)
        self.min_area_size = 100
        self.global_frame = 'odom'  # Could be 'map' if you have SLAM
        self.camera_frame = 'depth_link'
        self.visualisation = True

        # 2D bounding boxes of previously seen objects (to avoid double detection per frame)
        self.prev_bboxes = []

        # Camera models
        self.ccamera_model = None
        self.dcamera_model = None
        self.image_depth_ros = None
        self.color2depth_aspect = None

        # Topics (adjust if your actual robot differs)
        ccamera_info_topic = '/limo/depth_camera_link/camera_info'
        dcamera_info_topic = '/limo/depth_camera_link/depth/camera_info'
        cimage_topic       = '/limo/depth_camera_link/image_raw'
        dimage_topic       = '/limo/depth_camera_link/depth/image_raw'

        if self.real_robot:
            # Example real robot topics
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

        # Publisher for object location in 3D
        self.publisher_obj = self.create_publisher(
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
        """Compute ratio to align color image coords to depth image coords."""
        if (self.ccamera_model and self.dcamera_model and 
            self.color2depth_aspect is None):
            c_aspect = (math.atan2(self.ccamera_model.width, 
                                   2 * self.ccamera_model.fx()) / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 
                                   2 * self.dcamera_model.fx()) / self.dcamera_model.width)
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
        
        color_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")

        if self.real_robot:
            # Convert mm to meters
            depth_img /= 1000.0

        #----------------------------------------------------------------------
        # 1) Segment color blobs (red, green, blue)
        #----------------------------------------------------------------------
        red_mask   = cv2.inRange(color_img,  (0,   0,  80),  (50,  50, 255))
        green_mask = cv2.inRange(color_img,  (0,  80,   0),  (50, 255,  50))
        blue_mask  = cv2.inRange(color_img,  (80,  0,   0),  (255,  50,  50))
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #----------------------------------------------------------------------
        # 2) For each contour above min_area_size => bounding rect => check overlap => 3D
        #----------------------------------------------------------------------
        new_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_size:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rectA = Rectangle(x, y, x+w, y+h)

            # Check overlap with previously seen objects
            is_new = True
            for rectB in self.prev_bboxes:
                if rectA.overlaps(rectB):
                    is_new = False
                    break

            if not is_new:
                continue

            # If new, do 3D transform
            cx = x + w/2
            cy = y + h/2

            dx, dy = self.color_to_depth_coords(cx, cy, color_img, depth_img)
            if not (0 <= int(dx) < depth_img.shape[1] and 
                    0 <= int(dy) < depth_img.shape[0]):
                # out of depth image bounds
                continue

            depth_val = depth_img[int(dy), int(dx)]
            if depth_val <= 0.0 or math.isinf(depth_val) or math.isnan(depth_val):
                continue

            # Convert to 3D in camera frame
            camera_pose = self.pixel_to_3d_pose(cx, cy, depth_val)
            # TF to global frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame, self.camera_frame, rclpy.time.Time())
                global_pose = do_transform_pose(camera_pose, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            # Publish
            ps = PoseStamped(header=Header(frame_id=self.global_frame), pose=global_pose)
            self.publisher_obj.publish(ps)

            # Draw bounding box for visualization
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (255,255,0), 1)
            cv2.circle(color_img, (int(cx), int(cy)), 3, (255,0,255), -1)

            # Add to new bboxes
            new_bboxes.append(rectA)

        # Update prev_bboxes
        self.prev_bboxes = new_bboxes

        # Visualization
        if self.visualisation:
            disp_color = color_img.copy()
            disp_depth = depth_img.copy() / 10.0  # scale for viewing
            disp_color = cv2.resize(disp_color, (0,0), fx=0.5, fy=0.5)
            disp_depth = cv2.resize(disp_depth, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("combined_detector color", disp_color)
            cv2.imshow("combined_detector depth", disp_depth)
            cv2.waitKey(1)

    def color_to_depth_coords(self, cx, cy, color_img, depth_img):
        """Approx alignment of color pixel -> depth pixel using color2depth_aspect."""
        color_cx = color_img.shape[1] / 2.0
        color_cy = color_img.shape[0] / 2.0
        depth_cx = depth_img.shape[1] / 2.0
        depth_cy = depth_img.shape[0] / 2.0

        shift_x = cx - color_cx
        shift_y = cy - color_cy

        dx = depth_cx + shift_x * self.color2depth_aspect
        dy = depth_cy + shift_y * self.color2depth_aspect
        return dx, dy

    def pixel_to_3d_pose(self, cx, cy, depth_value):
        """
        Project a (cx, cy) pixel + depth into a 3D pose in the camera frame.
        """
        from image_geometry import PinholeCameraModel
        ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
        # scale ray so that Z = depth_value
        ray *= depth_value / ray[2]

        pose = Pose()
        pose.position.x = ray[0]
        pose.position.y = ray[1]
        pose.position.z = ray[2]
        pose.orientation.w = 1.0
        return pose

#------------------------------------------------------------------------------
# Counter Node (3D)
#   - Subscribes to /object_location (PoseStamped)
#   - Maintains a list of 3D positions, skipping any new detection within
#     'detection_threshold' distance of an existing object
#   - Logs count to CSV each time a **new** object is detected
#   - Prints updated total count to the console
#------------------------------------------------------------------------------
class Counter3D(Node):
    detection_threshold = 0.3  # in meters

    def __init__(self):
        super().__init__('counter_3d')
        self.detected_objects = []  # list of Pose (object positions)
        
        # Set up CSV logging
        self.csv_file = open('object_count_log.csv', 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['timestamp', 'count'])
        self.csv_writer.writeheader()
        
        self.subscriber = self.create_subscription(
            PoseStamped,
            '/object_location',
            self.counter_callback,
            qos_profile=qos.qos_profile_sensor_data)
        
        # Optional: publish a PoseArray for RViz
        self.publisher = self.create_publisher(
            PoseArray, '/object_count_array', qos.qos_profile_parameters)

        self.get_logger().info("Counter3D started. Waiting for object detections...")

    def counter_callback(self, msg: PoseStamped):
        new_pose = msg.pose
        # compare to existing detected_objects
        is_duplicate = False
        for existing_pose in self.detected_objects:
            dx = existing_pose.position.x - new_pose.position.x
            dy = existing_pose.position.y - new_pose.position.y
            dz = existing_pose.position.z - new_pose.position.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < self.detection_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            self.detected_objects.append(new_pose)

            # Write current time + new count to CSV
            # We can use either the ROS time stamp or node's clock
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.csv_writer.writerow({
                'timestamp': f'{stamp:.3f}',
                'count': len(self.detected_objects)
            })
            self.csv_file.flush()

        # Publish PoseArray
        parray = PoseArray(header=msg.header)
        parray.poses = self.detected_objects
        self.publisher.publish(parray)

        # Print current count in the console
        count_str = f'Total Unique Objects: {len(self.detected_objects)}'
        self.get_logger().info(count_str)

    def destroy_node(self):
        """Ensure we close our CSV file when the node is destroyed."""
        super().destroy_node()
        self.csv_file.close()

#------------------------------------------------------------------------------
# WarehouseSweeper Node
#   - Moves forward until near obstacle, brakes, turns left/right,
#   - Checks four corners of a nominal area,
#   - Returns to start once corners visited,
#   - Stops at start.
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
        
        # Laser data
        self.laser_ranges = []
        
        # Pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.start_x = None
        self.start_y = None
        
        # Example corners (adjust to your environment)
        self.corners = [
            (4.0,  4.0),
            (4.0, -4.0),
            (-4.0, -4.0),
            (-4.0,  4.0)
        ]
        self.visited_corners = [False]*len(self.corners)
        self.corner_tolerance = 1.0  # distance to corner

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
        self.dead_end_threshold = 0.7
        
        self.get_logger().info('WarehouseSweeper node started. Navigating...')

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_yaw = yaw
        
        # Record start pose if not set
        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f"Start pose recorded: ({self.start_x:.2f}, {self.start_y:.2f})")

    def timer_callback(self):
        if not self.laser_ranges:
            return
        
        # Check corners
        self.check_corners_visited()
        all_corners_visited = all(self.visited_corners)

        # If corners done, return home
        if all_corners_visited and self.state not in [STATE_RETURN, STATE_DONE, STATE_DEAD_END]:
            self.get_logger().info("All corners visited; heading back to start!")
            self.state = STATE_RETURN

        twist = Twist()

        # If in FORWARD/BRAKE/TURN, check for dead-end
        if self.state in [STATE_FORWARD, STATE_BRAKE, STATE_TURN]:
            if self.detect_dead_end():
                self.state = STATE_DEAD_END
                self.dead_end_time = 0
                self.dead_end_time_target = 20  # 2s at 10 Hz
                self.get_logger().info("Dead end detected; U-turning...")

        if self.state == STATE_FORWARD:
            front_dist = self.get_front_distance()
            if front_dist < self.obstacle_threshold:
                # brake
                self.state = STATE_BRAKE
                self.brake_time = 0
                self.brake_time_target = 5
            else:
                twist.linear.x = self.forward_speed

        elif self.state == STATE_BRAKE:
            # quick stop
            self.brake_time += 1
            if self.brake_time >= self.brake_time_target:
                self.decide_turn_direction()
                self.state = STATE_TURN
                self.turn_time = 0
                self.turn_time_target = 10
            # zero velocity while braking

        elif self.state == STATE_TURN:
            twist.angular.z = 0.5 * self.turn_dir
            self.turn_time += 1
            if self.turn_time >= self.turn_time_target:
                self.state = STATE_FORWARD

        elif self.state == STATE_DEAD_END:
            # Slow 180 turn
            twist.angular.z = 0.3
            self.dead_end_time += 1
            if self.dead_end_time >= self.dead_end_time_target:
                self.state = STATE_RETURN

        elif self.state == STATE_RETURN:
            # Move toward start_x, start_y
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            dist_to_start = math.sqrt(dx*dx + dy*dy)
            if dist_to_start < 0.5:
                self.get_logger().info("Reached start point!")
                self.state = STATE_DONE
            else:
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - self.current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                
                # Simple P-controller on yaw
                angular_speed = 1.0 * yaw_error
                angular_speed = max(-0.5, min(0.5, angular_speed))
                
                # if facing near the correct heading, move forward
                if abs(yaw_error) < 0.2:
                    twist.linear.x = 0.3
                twist.angular.z = angular_speed

        elif self.state == STATE_DONE:
            # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # We could shut down the node if we want
            # self.get_logger().info("Done Sweeping. Shutting down node.")
            # rclpy.shutdown()

        self.cmd_vel_pub.publish(twist)

    def decide_turn_direction(self):
        # Compare left vs. right distances to decide which way to turn
        right_d = self.get_right_distance()
        left_d = self.get_left_distance()
        # default right
        turn_dir = -1
        if right_d < 0.8 or right_d < left_d:
            turn_dir = 1
        self.turn_dir = turn_dir
        side_str = 'RIGHT' if turn_dir == -1 else 'LEFT'
        self.get_logger().info(f'Braking done. Turning {side_str}...')

    def detect_dead_end(self):
        front = self.get_front_distance()
        left  = self.get_left_distance()
        right = self.get_right_distance()
        if front < self.dead_end_threshold and left < self.dead_end_threshold and right < self.dead_end_threshold:
            return True
        return False

    def get_front_distance(self):
        mid = len(self.laser_ranges)//2
        return self.laser_ranges[mid] if self.laser_ranges else float('inf')

    def get_left_distance(self):
        mid = len(self.laser_ranges)//2
        idx = mid + 90
        idx = min(idx, len(self.laser_ranges)-1)
        return self.laser_ranges[idx] if self.laser_ranges else float('inf')

    def get_right_distance(self):
        mid = len(self.laser_ranges)//2
        idx = mid - 90
        idx = max(idx, 0)
        return self.laser_ranges[idx] if self.laser_ranges else float('inf')

    def check_corners_visited(self):
        for i, (cx, cy) in enumerate(self.corners):
            if not self.visited_corners[i]:
                dist = math.sqrt((cx - self.current_x)**2 + (cy - self.current_y)**2)
                if dist < self.corner_tolerance:
                    self.visited_corners[i] = True
                    self.get_logger().info(f"Visited corner {i+1} at ({cx:.1f}, {cy:.1f}).")

#------------------------------------------------------------------------------
# MAIN: Combine everything into one script using a MultiThreadedExecutor
#------------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    # Create our three nodes:
    movement_node = WarehouseSweeper()
    detector_node = CombinedDetector()
    counter_node  = Counter3D()

    # Spin them together so they run simultaneously
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(movement_node)
    executor.add_node(detector_node)
    executor.add_node(counter_node)

    try:
        executor.spin()
    finally:
        # Ensure files and nodes are closed properly
        movement_node.destroy_node()
        detector_node.destroy_node()
        counter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
