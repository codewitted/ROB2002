#!/usr/bin/env python3
"""
Robot Evaluation Script with:
  - AdvancedDetector
  - AdaptiveCoverage with 360° collision avoidance:
      • The robot follows a predetermined path (visit corners in order, then random coverage, then return home).
      • A 360° safe zone is enforced. If any obstacle is detected closer than 3× the robot’s characteristic distance
        (for example, for a robot of 0.5 m length, safe_distance = 1.5 m), the robot will turn away from the obstacle.
      • The robot then steers away (using a fixed lateral adjustment) until the safe zone is re‐established, thereby avoiding impact.
  - Changes in detected parameters or avoidance decisions) are printed to the terminal and logged.
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
        self.writer.writerow([
            "record_type", "time_sec", "color", "x", "y", "z",
            "scenario", "count_red", "count_green", "count_blue", "count_total", "extra_notes"
        ])
        self.start_time = time.time()
        self.scenario_name = scenario_name

    def log_detection(self, record_type, x, y, z, note=""):
        t_elapsed = time.time() - self.start_time
        self.writer.writerow([
            record_type, f"{t_elapsed:.2f}", "",
            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
            self.scenario_name, "", "", "", "", note
        ])
        self.file.flush()

    def log_summary(self, count_red, count_green, count_blue, note="End of run"):
        t_elapsed = time.time() - self.start_time
        total = count_red + count_green + count_blue
        self.writer.writerow([
            "SUMMARY", f"{t_elapsed:.2f}", "",
            "", "", "",
            self.scenario_name, count_red, count_green, count_blue, total, note
        ])
        self.file.flush()

    def close(self):
        self.file.close()


###############################################################################
# 2. ADVANCED DETECTOR NODE (unchanged from prior example)
###############################################################################
class AdvancedDetector(Node):
    def __init__(self, scenario_name="default_scenario"):
        super().__init__('advanced_detector')
        self.logger = CSVLogger(scenario_name=scenario_name)
        # TF & frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'odom'
        self.camera_frame = 'depth_link'
        # Sensor models
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None
        self.bridge = CvBridge()
        self.depth_image_ros = None
        self.real_robot = False
        self.detected_objects = []
        self.detection_threshold = 0.6
        # HSV thresholds for red, green, blue
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)
        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)
        # LiDAR subscription
        self.lidar_ranges = []
        self.create_subscription(LaserScan, 'scan', self.lidar_callback, qos.qos_profile_sensor_data)
        # Camera info and image subscriptions
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
            c_aspect = (math.atan2(self.ccamera_model.width, 2*self.ccamera_model.fx()) / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2*self.dcamera_model.fx()) / self.dcamera_model.width)
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
            depth_img /= 1000.0  # mm->m
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
            roi_r = red_mask[y:y+h, x:x+w]
            roi_g = green_mask[y:y+h, x:x+w]
            roi_b = blue_mask[y:y+h, x:x+w]
            sum_r = np.sum(roi_r)
            sum_g = np.sum(roi_g)
            sum_b = np.sum(roi_b)
            color_label = "red"
            max_sum = sum_r
            if sum_g > max_sum:
                color_label = "green"
                max_sum = sum_g
            if sum_b > max_sum:
                color_label = "blue"
                max_sum = sum_b
            cx = x + w/2
            cy = y + h/2
            dx, dy = self.color2depth_coords(cx, cy, color_img, depth_img)
            if (dx < 0 or dy < 0 or dx >= depth_img.shape[1] or dy >= depth_img.shape[0]):
                continue
            zval = depth_img[int(dy), int(dx)]
            if zval <= 0.0 or math.isnan(zval) or math.isinf(zval):
                continue
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
                cv2.rectangle(color_img, (x, y), (x+w, y+h), (255,255,0), 2)
                cv2.putText(color_img, color_label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        if self.visualise:
            cv2.imshow("AdvancedDetector - Color", color_img)
            scaled_depth = depth_img / 5.0
            cv2.imshow("AdvancedDetector - Depth", scaled_depth)
            cv2.waitKey(1)

    def overlaps(self, x1, y1, w1, h1, x2, y2, w2, h2):
        if (x1 > x2+w2 or x2 > x1+w1 or y1 > y2+h2 or y2 > y1+h1):
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
                if math.sqrt(dx*dx + dy*dy + dz*dz) < self.detection_threshold:
                    return False
        return True

    def finish_and_log_summary(self, note="End of run"):
        red_count   = sum(1 for o in self.detected_objects if o['color'] == 'red')
        green_count = sum(1 for o in self.detected_objects if o['color'] == 'green')
        blue_count  = sum(1 for o in self.detected_objects if o['color'] == 'blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()


###############################################################################
# 3. ADAPTIVE COVERAGE with BASIC CAR MOVEMENT & OBSTACLE AVOIDANCE
###############################################################################
class AdaptiveCoverage(Node):
    """
    - The robot follows a fixed sequence:
        1. Visit four corners in order.
        2. Then perform a short random coverage phase.
        3. Then return home.
    - While traversing, if an obstacle is detected within a safe distance (set here to 2× the robot's length)
      in any direction, the robot does NOT change its overall course.
      Instead, it steers laterally (adds a fixed angular adjustment) to avoid impact.
    - When the path ahead is clear (readings above a brake threshold) the robot accelerates at maximum speed.
    - Only new (changed) messages are printed to the terminal and logged.
    """
    def __init__(self):
        super().__init__('adaptive_coverage')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.laser_ranges = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Task state: VISIT_CORNERS, RANDOM_COVERAGE, RETURN_HOME.
        self.state = "VISIT_CORNERS"
        self.corner_idx = 0
        self.done = False

        # Pre-defined corners.
        self.corners = [
            (4.5,  4.5),
            (4.5, -4.5),
            (-4.5, -4.5),
            (-4.5,  4.5)
        ]
        self.corner_tolerance = 0.7

        self.random_targets = []
        self.rng = random.Random()

        self.start_x = None
        self.start_y = None

        # Speed parameters.
        self.fast_speed = 0.5
        self.slow_speed = 0.15
        self.angular_speed = 0.3

        # New safety parameters:
        # Set safe distance to 2× the robot’s length. (For example, if the robot’s length is ~0.5 m, safe_distance = 1.0 m.)
        self.safe_distance = 1.0
        # But now, to "turn away from the wall and avoid impact" we use the following behavior:
        # If any obstacle is detected within 3× the robot’s length (e.g. 3×0.5 = 1.5 m) anywhere 360° around,
        # then the robot will initiate a lateral turn away from it.
        self.impact_distance = 1.5
        # The brake_distance (for accelerating when the path is clear) remains:
        self.brake_distance = 2.0

        # For basic avoidance, we compute a fixed lateral adjustment if the minimum reading
        # in either the left or right half of the laser scan is below the safe distance.
        self.last_avoid_msg = ""
        self.last_log_msg = ""

        self.obstacle_logger = CSVLogger(scenario_name="AdaptiveCoverage_Obstacles")
        self.get_logger().info("AdaptiveCoverage node ready (basic car movement with 360° obstacle avoidance).")

    def scan_cb(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_x = px
        self.current_y = py
        self.current_yaw = yaw
        if self.start_x is None:
            self.start_x = px
            self.start_y = py
            self._print_once(f"Start position set: ({px:.2f}, {py:.2f})")

    def get_current_target(self):
        if self.state == "VISIT_CORNERS" and self.corner_idx < len(self.corners):
            return self.corners[self.corner_idx]
        elif self.state == "RANDOM_COVERAGE" and self.random_targets:
            return self.random_targets[0]
        elif self.state == "RETURN_HOME":
            return (self.start_x, self.start_y)
        return None

    def timer_cb(self):
        if self.done:
            self.cmd_vel_pub.publish(Twist())
            return
        if not self.laser_ranges:
            return

        target = self.get_current_target()
        if target is None:
            return
        tx, ty = target
        desired_heading = math.atan2(ty - self.current_y, tx - self.current_x)
        heading_error = desired_heading - self.current_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # First: if any reading (360°) is below the impact threshold, turn away.
        if min(self.laser_ranges) < self.impact_distance:
            # Determine which side (left/right) is closer
            total = len(self.laser_ranges)
            mid = total // 2
            left_min = min(self.laser_ranges[mid:]) if total - mid > 0 else float('inf')
            right_min = min(self.laser_ranges[:mid]) if mid > 0 else float('inf')
            if left_min < right_min:
                # Obstacle on left: steer right.
                avoidance_offset = -0.4
                msg = f"Impact alert: obstacle on left at {left_min:.2f} m – turning right."
            else:
                # Obstacle on right: steer left.
                avoidance_offset = 0.4
                msg = f"Impact alert: obstacle on right at {right_min:.2f} m – turning left."
            self._print_once(msg)
            self.obstacle_logger.log_detection("impact", self.current_x, self.current_y, 0, note=msg)
        else:
            # Otherwise, compute an avoidance offset (based on the left/right halves of laser-scan)
            avoidance_offset = self._get_avoidance_adjustment()

        total_angular = heading_error + avoidance_offset

        # Decide on speed: if the front sector is clear (above brake_distance) then use fast_speed
        front_clear = self._front_clearance() > self.brake_distance
        linear_speed = self.fast_speed if front_clear else self.slow_speed

        tw = Twist()
        tw.linear.x = linear_speed
        tw.angular.z = max(-self.angular_speed, min(self.angular_speed, total_angular))
        self.cmd_vel_pub.publish(tw)

        # Check if the target is reached.
        if math.hypot(tx - self.current_x, ty - self.current_y) < self.corner_tolerance:
            self.cmd_vel_pub.publish(Twist())
            if self.state == "VISIT_CORNERS":
                self._print_once(f"Corner {self.corner_idx+1} reached.")
                self.corner_idx += 1
                if self.corner_idx >= len(self.corners):
                    self._print_once("All 4 corners reached. Starting random coverage.")
                    for _ in range(5):
                        rx = self.rng.uniform(-4.5, 4.5)
                        ry = self.rng.uniform(-4.5, 4.5)
                        self.random_targets.append((rx, ry))
                    self.state = "RANDOM_COVERAGE"
            elif self.state == "RANDOM_COVERAGE" and not self.random_targets:
                self._print_once("Random coverage complete. Returning home.")
                self.state = "RETURN_HOME"
            elif self.state == "RETURN_HOME":
                self._print_once("Reached home. Task complete.")
                self.done = True

    def _get_avoidance_adjustment(self):
        """
        Examine the left and right halves of the laser-scan.
        If an obstacle in one half is detected within the safe distance (2× robot length)
        then return a fixed angular offset: positive to steer left, negative to steer right.
        """
        if not self.laser_ranges:
            return 0.0
        total = len(self.laser_ranges)
        mid = total // 2
        left_section = self.laser_ranges[mid:]
        right_section = self.laser_ranges[:mid]

        left_min = min(left_section) if left_section else float('inf')
        right_min = min(right_section) if right_section else float('inf')
        adjustment = 0.0

        if left_min < self.safe_distance or right_min < self.safe_distance:
            if left_min < right_min:
                adjustment = -0.3  # steer right
                msg = f"Obstacle closer on left (min {left_min:.2f} m): steering right."
            else:
                adjustment = 0.3   # steer left
                msg = f"Obstacle closer on right (min {right_min:.2f} m): steering left."
            self._print_once(msg)
            self.obstacle_logger.log_detection("avoid", self.current_x, self.current_y, 0, note=msg)
        else:
            msg = "No lateral avoidance required; path clear."
            self._print_once(msg)
            self.obstacle_logger.log_detection("clear", self.current_x, self.current_y, 0, note=msg)
        return adjustment

    def _front_clearance(self):
        """Return the minimum distance measured in a narrow forward sector."""
        total = len(self.laser_ranges)
        mid = total // 2
        idx_start = max(0, mid - 5)
        idx_end = min(total, mid + 5)
        sector = self.laser_ranges[idx_start:idx_end]
        if not sector:
            return float('inf')
        return min(sector)

    def _print_once(self, msg):
        """Print and log the message only if it differs from the previous message."""
        if msg != self.last_log_msg:
            self.get_logger().info(msg)
            self.last_log_msg = msg


###############################################################################
# 4. MAIN: Launch Detector + Coverage
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    scenario_name = "reversing_demo"
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
        coverage.obstacle_logger.close()
        detector.destroy_node()
        coverage.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
