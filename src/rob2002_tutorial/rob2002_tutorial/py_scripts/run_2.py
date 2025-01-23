#!/usr/bin/env python3
"""
Robot Evaluation Script with:
  - Advanced Detector (with CSV logging).
  - Adaptive Coverage with Direction Blocking based on LaserScan data.
  - Safe distance enforcement (4x robot length) and obstacle avoidance.
  
Once a direction corresponding to a corner (or any target) is blocked by an obstacle,
that direction is recorded and the robot skips it. The robot stops after completing
the navigation tasks.
"""

import math
import time
import random
import datetime
import csv
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge

###############################################################################
# 1. CSV LOGGER – For Recording Results
###############################################################################
class CSVLogger:
    def __init__(self, scenario_name="default_scenario"):
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"results_{scenario_name}_{now_str}.csv"
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)

        # CSV header
        self.writer.writerow([
            "record_type",
            "time_sec",
            "color",
            "x", "y", "z",
            "scenario",
            "count_red",
            "count_green",
            "count_blue",
            "count_total",
            "extra_notes",
        ])

        self.start_time = time.time()
        self.scenario_name = scenario_name

    def log_detection(self, color_label, x, y, z, note=""):
        t_elapsed = time.time() - self.start_time
        self.writer.writerow([
            "DETECTION",
            f"{t_elapsed:.2f}",
            color_label,
            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
            self.scenario_name,
            "", "", "", "",
            note
        ])
        self.file.flush()

    def log_summary(self, count_red, count_green, count_blue, note=""):
        t_elapsed = time.time() - self.start_time
        total = count_red + count_green + count_blue
        self.writer.writerow([
            "SUMMARY",
            f"{t_elapsed:.2f}",
            "",
            "", "", "",
            self.scenario_name,
            count_red,
            count_green,
            count_blue,
            total,
            note
        ])
        self.file.flush()

    def close(self):
        self.file.close()


###############################################################################
# 2. ADVANCED DETECTOR NODE
###############################################################################
class AdvancedDetector(Node):
    def __init__(self, scenario_name="default_scenario"):
        super().__init__('advanced_detector')
        self.logger = CSVLogger(scenario_name=scenario_name)

        # TF / frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'odom'
        self.camera_frame = 'depth_link'

        # Sensor models and CV bridge
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None
        self.bridge = CvBridge()
        self.depth_image_ros = None
        self.real_robot = False

        # Storage for unique detections
        self.detected_objects = []
        self.detection_threshold = 0.6

        # HSV thresholds for colors
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)
        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)

        # LiDAR subscription and placeholder
        self.lidar_ranges = []
        self.create_subscription(LaserScan, 'scan', self.lidar_callback, qos.qos_profile_sensor_data)

        # Camera topics and subscriptions
        ccamera_info_topic = '/limo/depth_camera_link/camera_info'
        dcamera_info_topic = '/limo/depth_camera_link/depth/camera_info'
        cimage_topic       = '/limo/depth_camera_link/image_raw'
        dimage_topic       = '/limo/depth_camera_link/depth/image_raw'
        if self.real_robot:
            ccamera_info_topic = '/camera/color/camera_info'
            dcamera_info_topic = '/camera/depth/camera_info'
            cimage_topic       = '/camera/color/image_raw'
            dimage_topic       = '/camera/depth/image_raw'
            self.camera_frame  = 'camera_color_optical_frame'
        self.create_subscription(CameraInfo, ccamera_info_topic, self.ccamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(CameraInfo, dcamera_info_topic, self.dcamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(Image, cimage_topic, self.image_color_callback, qos.qos_profile_sensor_data)
        self.create_subscription(Image, dimage_topic, self.image_depth_callback, qos.qos_profile_sensor_data)

        self.visualise = True
        self.get_logger().info("AdvancedDetector initialized (with CSV logging).")

    def lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = list(msg.ranges)

    def ccamera_info_callback(self, msg: CameraInfo):
        if not self.ccamera_model:
            from image_geometry import PinholeCameraModel
            self.ccamera_model = PinholeCameraModel()
            self.ccamera_model.fromCameraInfo(msg)
            self.update_color2depth_aspect()

    def dcamera_info_callback(self, msg: CameraInfo):
        if not self.dcamera_model:
            from image_geometry import PinholeCameraModel
            self.dcamera_model = PinholeCameraModel()
            self.dcamera_model.fromCameraInfo(msg)
            self.update_color2depth_aspect()

    def update_color2depth_aspect(self):
        if (self.ccamera_model and self.dcamera_model and self.color2depth_aspect is None):
            c_aspect = (math.atan2(self.ccamera_model.width, 2 * self.ccamera_model.fx()) / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2 * self.dcamera_model.fx()) / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(f"color2depth_aspect = {self.color2depth_aspect:.3f}")

    def image_depth_callback(self, msg: Image):
        self.depth_image_ros = msg

    def image_color_callback(self, msg: Image):
        if (self.color2depth_aspect is None or self.depth_image_ros is None or
            not self.ccamera_model or not self.dcamera_model):
            return

        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.depth_image_ros, "32FC1")
        if self.real_robot:
            depth_img /= 1000.0  # Convert mm to m if needed

        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask  = cv2.inRange(hsv, self.lower_blue,  self.upper_blue)

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame_bboxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if any(self.overlaps(x, y, w, h, px, py, pw, ph) for (px, py, pw, ph) in frame_bboxes):
                continue
            frame_bboxes.append((x, y, w, h))
            # Color determination
            roi_r = red_mask[y:y+h, x:x+w]
            roi_g = green_mask[y:y+h, x:x+w]
            roi_b = blue_mask[y:y+h, x:x+w]
            sum_r = np.sum(roi_r)
            sum_g = np.sum(roi_g)
            sum_b = np.sum(roi_b)
            color_label = 'red'
            max_sum = sum_r
            if sum_g > max_sum:
                color_label = 'green'
                max_sum = sum_g
            if sum_b > max_sum:
                color_label = 'blue'
                max_sum = sum_b
            cx = x + w/2
            cy = y + h/2
            dx, dy = self.color2depth_coords(cx, cy, color_img, depth_img)
            if (dx < 0 or dy < 0 or dx >= depth_img.shape[1] or dy >= depth_img.shape[0]):
                continue
            zval = depth_img[int(dy), int(dx)]
            if zval <= 0.0 or math.isnan(zval) or math.isinf(zval):
                continue

            # Compute 3D coordinates using the camera model
            from image_geometry import PinholeCameraModel
            ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
            ray *= (zval / ray[2])
            camera_pose = Pose()
            camera_pose.position.x = ray[0]
            camera_pose.position.y = ray[1]
            camera_pose.position.z = ray[2]
            camera_pose.orientation.w = 1.0

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame, self.camera_frame, rclpy.time.Time())
                global_pose = do_transform_pose(camera_pose, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            gx = global_pose.position.x
            gy = global_pose.position.y
            gz = global_pose.position.z

            if not self.is_new_object(gx, gy, gz, color_label):
                continue

            self.detected_objects.append({'color': color_label, 'x': gx, 'y': gy, 'z': gz})
            self.logger.log_detection(color_label, gx, gy, gz)

            if self.visualise:
                cv2.rectangle(color_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(color_img, color_label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if self.visualise:
            cv2.imshow("AdvancedDetector - Color", color_img)
            scaled_depth = depth_img / 5.0
            cv2.imshow("AdvancedDetector - Depth", scaled_depth)
            cv2.waitKey(1)

    def overlaps(self, x1, y1, w1, h1, x2, y2, w2, h2):
        if (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1):
            return False
        return True

    def color2depth_coords(self, cx, cy, color_img, depth_img):
        c_cx = color_img.shape[1] / 2.0
        c_cy = color_img.shape[0] / 2.0
        d_cx = depth_img.shape[1] / 2.0
        d_cy = depth_img.shape[0] / 2.0

        shift_x = cx - c_cx
        shift_y = cy - c_cy
        dx = d_cx + shift_x * self.color2depth_aspect
        dy = d_cy + shift_y * self.color2depth_aspect
        return dx, dy

    def is_new_object(self, gx, gy, gz, color_label):
        for obj in self.detected_objects:
            if obj['color'] == color_label:
                dx = obj['x'] - gx
                dy = obj['y'] - gy
                dz = obj['z'] - gz
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < self.detection_threshold:
                    return False
        return True

    def finish_and_log_summary(self, note="End of run"):
        red_count = sum(1 for o in self.detected_objects if o['color'] == 'red')
        green_count = sum(1 for o in self.detected_objects if o['color'] == 'green')
        blue_count = sum(1 for o in self.detected_objects if o['color'] == 'blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()


###############################################################################
# 3. ADAPTIVE COVERAGE with DIRECTION BLOCKING
###############################################################################
class AdaptiveCoverage(Node):
    """
    This node navigates to predefined corners while maintaining a safe distance (4x the robot's length)
    from obstacles. The robot divides its 360° surroundings into sectors and marks a sector as blocked if
    an obstacle is detected within the safe clearance. If a target lies in a blocked direction, the target
    is skipped.
    """
    def __init__(self):
        super().__init__('adaptive_coverage')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_cb, qos.qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Laser and pose variables
        self.laser_ranges = []
        self.laser_msg = None    # To store the latest LaserScan message
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Navigation state
        self.state = "NAVIGATING"
        self.current_target_idx = 0
        self.done = False

        # Warehouse corners (targets)
        self.corners = [
            (4.5, 4.5),
            (4.5, -4.5),
            (-4.5, -4.5),
            (-4.5, 4.5),
        ]
        self.corner_tolerance = 0.7

        # Sectors for 360° blocked directions
        self.num_sectors = 12
        self.sector_size = 360 / self.num_sectors  # degrees per sector
        self.blocked_sectors = set()

        # Robot parameters
        self.robot_length = 0.5     # in meters (example value)
        self.safe_distance = 4 * self.robot_length

        # Speed and distance parameters
        self.fast_speed = 0.5
        self.slow_speed = 0.15
        self.angular_speed = 0.3
        self.min_clearance = self.safe_distance   # obstacles within this distance mark a sector as blocked
        self.brake_distance = self.safe_distance * 0.25

        self.get_logger().info("AdaptiveCoverage node ready (with direction blocking).")

    def scan_cb(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)
        self.laser_msg = msg  # Store the full LaserScan message for angle_min and angle_max

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_x = px
        self.current_y = py
        self.current_yaw = yaw
        if not hasattr(self, 'start_x') or self.start_x is None:
            self.start_x = px
            self.start_y = py
            self.get_logger().info(f"Start position set: ({px:.2f}, {py:.2f})")

    def update_blocked_sectors(self):
        """
        Update blocked sectors based on LaserScan data. A sector is blocked if a reading in that sector is below min_clearance.
        """
        if self.laser_msg is None:
            return

        # Create an array of angles for each laser reading based on the stored LaserScan message.
        angles = np.linspace(self.laser_msg.angle_min, self.laser_msg.angle_max, len(self.laser_ranges))
        sector_blocked = set()

        for i in range(self.num_sectors):
            angle_min_sector = math.radians(i * self.sector_size)
            angle_max_sector = math.radians((i + 1) * self.sector_size)
            # Get indices of readings whose angle lies within this sector.
            indices = [j for j, angle in enumerate(angles) if angle_min_sector <= angle < angle_max_sector]
            if not indices:
                continue
            dists = [self.laser_ranges[j] for j in indices
                     if not math.isinf(self.laser_ranges[j]) and not math.isnan(self.laser_ranges[j])]
            min_dist = min(dists, default=float('inf'))
            if min_dist < self.min_clearance:
                sector_blocked.add(i)
        self.blocked_sectors = sector_blocked
        if self.blocked_sectors:
            self.get_logger().debug(f"Blocked sectors: {self.blocked_sectors}")
        else:
            self.get_logger().debug("No sectors are currently blocked.")

    def is_direction_blocked(self, target):
        """
        Check if the direction to a target falls within a blocked sector.
        """
        dx = target[0] - self.current_x
        dy = target[1] - self.current_y
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        sector = int(target_angle // self.sector_size)
        return sector in self.blocked_sectors

    def record_blocked_direction(self, target):
        self.get_logger().info(f"Recorded blocked direction toward target: {target}")

    def navigate_to_target(self, target):
        tx, ty = target
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < self.corner_tolerance:
            # Target reached
            self.cmd_vel_pub.publish(Twist())
            return True

        front_dist = self.get_front_distance()
        if front_dist < self.min_clearance:
            self.get_logger().warn("Too close to obstacle! Blocking current direction.")
            self.record_blocked_direction(target)
            self.current_target_idx += 1
            return False

        desired_yaw = math.atan2(dy, dx)
        yaw_err = desired_yaw - self.current_yaw
        # Normalize yaw error to [-pi, pi]
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        tw = Twist()
        if front_dist < self.brake_distance:
            tw.linear.x = self.slow_speed
        else:
            tw.linear.x = self.fast_speed
        tw.angular.z = max(-self.angular_speed, min(self.angular_speed, yaw_err))
        self.cmd_vel_pub.publish(tw)
        return False

    def get_front_distance(self):
        if not self.laser_ranges:
            return float('inf')
        front_idx = len(self.laser_ranges) // 2
        front_distance = self.laser_ranges[front_idx]
        if math.isinf(front_distance) or math.isnan(front_distance):
            return float('inf')
        return front_distance

    def timer_cb(self):
        if self.done:
            self.cmd_vel_pub.publish(Twist())
            return

        self.update_blocked_sectors()

        if self.state == "NAVIGATING":
            if self.current_target_idx < len(self.corners):
                target = self.corners[self.current_target_idx]
                if self.is_direction_blocked(target):
                    self.get_logger().warn(f"Direction to target {target} is blocked. Skipping.")
                    self.record_blocked_direction(target)
                    self.current_target_idx += 1
                    return
                reached = self.navigate_to_target(target)
                if reached:
                    self.get_logger().info(f"Corner {self.current_target_idx + 1} reached.")
                    self.current_target_idx += 1
            else:
                self.get_logger().info("All corners visited or blocked. Stopping.")
                self.done = True
                self.shutdown_robot()

    def shutdown_robot(self):
        tw = Twist()
        self.cmd_vel_pub.publish(tw)
        self.get_logger().info("Robot has been stopped.")

    def destroy_node(self):
        if not self.done:
            self.done = True
            self.shutdown_robot()
        super().destroy_node()


###############################################################################
# 4. MAIN: Launch Detector + Coverage
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    scenario_name = "direction_blocking_demo"

    detector = AdvancedDetector(scenario_name=scenario_name)
    coverage = AdaptiveCoverage()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    executor.add_node(coverage)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detector.finish_and_log_summary(note="Run completed or interrupted.")
        detector.destroy_node()
        coverage.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
