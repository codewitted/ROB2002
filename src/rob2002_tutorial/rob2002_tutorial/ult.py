#!/usr/bin/env python3
"""
Improved Robot Evaluation Script with:
  - Sensor Fusion placeholders (RGB + Depth + LiDAR).
  - Adaptive coverage planning (a simple "hybrid" of corner-visits + random coverage).
  - Temporal data integration placeholders (multi-angle checking).
  - Learning-based detection placeholder (CNN usage).
  - Sim2Real bridging notes (domain randomization, etc.).
  - Obstacle-based speed control (fast if clear, brake & slow if obstacle).
  - 4-corner coverage requirement.
  - CSV logging for detections and final summaries.

Author: ChatGPT
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

from geometry_msgs.msg import Twist, Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge


###############################################################################
# 1. CSV LOGGER – For Recording All Results (color, location, time, etc.)
###############################################################################
class CSVLogger:
    """
    Creates a CSV file to record detections and final summaries.
    Each detection: 'DETECTION' + time + color + (x,y,z).
    Final summary:  'SUMMARY' + time + scenario + total counts, etc.
    """
    def __init__(self, scenario_name="default_scenario"):
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"results_{scenario_name}_{now_str}.csv"

        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)

        # CSV header
        self.writer.writerow([
            "record_type",       # DETECTION or SUMMARY
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
        """
        Write a DETECTION row.
        """
        t_elapsed = time.time() - self.start_time
        self.writer.writerow([
            "DETECTION",
            f"{t_elapsed:.2f}",
            color_label,
            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
            self.scenario_name,
            "", "", "", "",  # no counts in DETECTION rows
            note
        ])
        self.file.flush()

    def log_summary(self, count_red, count_green, count_blue, note=""):
        """
        Write a SUMMARY row.
        """
        t_elapsed = time.time() - self.start_time
        total = count_red + count_green + count_blue
        self.writer.writerow([
            "SUMMARY",
            f"{t_elapsed:.2f}",
            "",   # no color here
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
# 2. ADVANCED DETECTOR NODE (Placeholders for Sensor Fusion, ML, etc.)
###############################################################################
class AdvancedDetector(Node):
    """
    An improved detection node that:
      - Uses color + depth + optional LiDAR data to reduce false positives (sensor fusion).
      - Has placeholders for advanced (CNN) detection.
      - Applies temporal integration (a simplistic approach) to avoid re-counting.
      - Logs results to CSV.
    """
    def __init__(self, scenario_name="default_scenario"):
        super().__init__('advanced_detector')

        # CSV Logger
        self.logger = CSVLogger(scenario_name=scenario_name)

        # TF / frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'odom'    # or 'map', depending on your setup
        self.camera_frame = 'depth_link'

        # Sensor models
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None

        # Basic image / depth
        self.bridge = CvBridge()
        self.depth_image_ros = None

        # Whether we run on real hardware (changes topics, mm->m, etc.)
        self.real_robot = False

        # Storage for unique detections (temporal integration)
        self.detected_objects = []  # each is a dict {color, x, y, z}
        self.detection_threshold = 0.6  # 3D radius to treat "same object"

        # HSV thresholds – placeholder
        # (Better: a learned color segmentation or CNN in real code)
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)

        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)

        # Example: placeholders for LiDAR data for sensor fusion
        self.lidar_ranges = []
        self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos.qos_profile_sensor_data
        )

        # Camera info
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

        # Subscriptions
        self.create_subscription(
            CameraInfo, ccamera_info_topic,
            self.ccamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, dcamera_info_topic,
            self.dcamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(
            Image, cimage_topic,
            self.image_color_callback, qos.qos_profile_sensor_data)
        self.create_subscription(
            Image, dimage_topic,
            self.image_depth_callback, qos.qos_profile_sensor_data)

        # Visualization toggle
        self.visualise = True

        self.get_logger().info("AdvancedDetector initialized (with CSV logging).")

    def lidar_callback(self, msg: LaserScan):
        """
        Example placeholder: In a real sensor‐fusion pipeline, you might combine
        LiDAR data with color+depth to reduce false positives or confirm range.
        """
        self.lidar_ranges = list(msg.ranges)
        # For now, not used heavily. But could be used to confirm an object's approximate distance.

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
        if (self.ccamera_model and 
            self.dcamera_model and 
            self.color2depth_aspect is None):
            c_aspect = (math.atan2(self.ccamera_model.width, 2*self.ccamera_model.fx())
                        / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2*self.dcamera_model.fx())
                        / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(
                f"color2depth_aspect = {self.color2depth_aspect:.3f}")

    def image_depth_callback(self, msg: Image):
        self.depth_image_ros = msg

    def image_color_callback(self, msg: Image):
        if (self.color2depth_aspect is None or
            self.depth_image_ros is None or
            not self.ccamera_model or
            not self.dcamera_model):
            return

        # Convert to OpenCV
        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.depth_image_ros, "32FC1")
        if self.real_robot:
            depth_img /= 1000.0  # mm->m

        # Convert to HSV
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # Multi-interval for Red
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask  = cv2.inRange(hsv, self.lower_blue,  self.upper_blue)

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        contours, _   = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Very simplistic CNN placeholder:
        # In a real system, you might do:
        #   bounding box -> cropped region -> run through a trained model
        #   -> decide color or even shape. We skip that detail here.

        # Intra-frame anti-double counting
        frame_bboxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100.0:  # min area
                continue

            x, y, w, h = cv2.boundingRect(c)
            # Simple bounding box overlap check with frame_bboxes
            if any(self.overlaps(x, y, w, h, px, py, pw, ph) for (px, py, pw, ph) in frame_bboxes):
                continue
            frame_bboxes.append((x, y, w, h))

            # Decide color by sum of mask within the bounding box
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

            # Depth coords
            cx = x + w/2
            cy = y + h/2
            dx, dy = self.color2depth_coords(cx, cy, color_img, depth_img)
            if (dx < 0 or dy < 0 or
                dx >= depth_img.shape[1] or
                dy >= depth_img.shape[0]):
                continue

            zval = depth_img[int(dy), int(dx)]
            if zval <= 0.0 or math.isnan(zval) or math.isinf(zval):
                continue

            # 3D projection
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

            # Temporal integration: skip if we’ve counted an object of same color
            # within self.detection_threshold in 3D
            if not self.is_new_object(gx, gy, gz, color_label):
                continue

            # Record new object
            self.detected_objects.append({
                'color': color_label,
                'x': gx, 'y': gy, 'z': gz
            })
            # Log in CSV
            self.logger.log_detection(color_label, gx, gy, gz)

            # Optional: Draw on the color image
            if self.visualise:
                cv2.rectangle(color_img, (x,y), (x+w, y+h), (255,255,0), 2)
                cv2.putText(color_img, color_label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Show images
        if self.visualise:
            cv2.imshow("AdvancedDetector - Color", color_img)
            scaled_depth = depth_img / 5.0
            cv2.imshow("AdvancedDetector - Depth", scaled_depth)
            cv2.waitKey(1)

    def overlaps(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Simple bounding box overlap check."""
        if (x1 > x2+w2 or x2 > x1+w1 or
            y1 > y2+h2 or y2 > y1+h1):
            return False
        return True

    def color2depth_coords(self, cx, cy, color_img, depth_img):
        """Map color pixel coords to approximate depth coords."""
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
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < self.detection_threshold:
                    return False
        return True

    def finish_and_log_summary(self, note="End of run"):
        """Call this when the robot is done: logs final summary of detections."""
        red_count   = sum(1 for o in self.detected_objects if o['color']=='red')
        green_count = sum(1 for o in self.detected_objects if o['color']=='green')
        blue_count  = sum(1 for o in self.detected_objects if o['color']=='blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()


###############################################################################
# 3. ADAPTIVE COVERAGE & OBSTACLE‐AWARE NAVIGATION
###############################################################################
class AdaptiveCoverage(Node):
    """
    A node that covers 4 corners of a warehouse, plus random expansions,
    and adjusts speed based on obstacle proximity:
      - If clear ahead => move fast
      - If obstacle => brake and slow maneuver
    Also tries not to get too close to walls or boxes (using LaserScan).
    """
    def __init__(self):
        super().__init__('adaptive_coverage')

        # Movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_cb, 10)

        # Key internal states
        self.laser_ranges = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.state = "VISIT_CORNERS"
        self.corner_idx = 0
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Corners of the warehouse (example ~10x10)
        self.corners = [
            ( 4.5,  4.5),
            ( 4.5, -4.5),
            (-4.5, -4.5),
            (-4.5,  4.5),
        ]
        self.corner_tolerance = 0.7

        # After corners visited, do random coverage
        self.random_targets = []
        self.rng = random.Random()

        # Speeds
        self.fast_speed = 0.5
        self.slow_speed = 0.15
        self.angular_speed = 0.3

        # Obstacle thresholds
        self.brake_distance = 1.0  # If anything <1.0m ahead => slow or stop
        self.min_clearance  = 0.5  # desired distance from walls/boxes

        # Return-home logic after random coverage
        self.start_x = None
        self.start_y = None
        self.done = False

        self.get_logger().info("AdaptiveCoverage node ready.")

    def scan_cb(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_cb(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_yaw = yaw

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f"Start position = ({self.start_x:.2f}, {self.start_y:.2f})")

    def timer_cb(self):
        if self.done:
            self.cmd_vel_pub.publish(Twist())  # stop
            return

        if not self.laser_ranges:
            return

        # If we haven't visited corners, proceed corner by corner
        if self.state == "VISIT_CORNERS":
            if self.corner_idx < len(self.corners):
                target = self.corners[self.corner_idx]
                if self.navigate_to_target(target):
                    self.get_logger().info(f"Corner {self.corner_idx+1} reached.")
                    self.corner_idx += 1
            else:
                self.get_logger().info("All corners visited. Switch to random coverage.")
                # generate random waypoints
                for _ in range(5):  # e.g. 5 random targets
                    rx = self.rng.uniform(-4.5, 4.5)
                    ry = self.rng.uniform(-4.5, 4.5)
                    self.random_targets.append((rx, ry))
                self.state = "RANDOM_COVERAGE"

        elif self.state == "RANDOM_COVERAGE":
            if self.random_targets:
                target = self.random_targets[0]
                if self.navigate_to_target(target):
                    self.random_targets.pop(0)
            else:
                self.get_logger().info("Random coverage done, returning home.")
                self.state = "RETURN_HOME"

        elif self.state == "RETURN_HOME":
            home_target = (self.start_x, self.start_y)
            reached = self.navigate_to_target(home_target)
            if reached:
                self.get_logger().info("Robot is back at start. DONE.")
                self.done = True

    def navigate_to_target(self, target):
        """
        Move towards 'target' (x, y) with obstacle-based speed control.
        Return True if within tolerance, else False.
        """
        tx, ty = target
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < self.corner_tolerance:
            # Stop
            tw = Twist()
            self.cmd_vel_pub.publish(tw)
            return True

        # Heading
        desired_yaw = math.atan2(dy, dx)
        yaw_err = desired_yaw - self.current_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        # Speed control
        front_dist = self.get_front_distance()
        if front_dist < self.brake_distance:
            # Slow or stop
            tw = Twist()
            tw.linear.x = self.slow_speed
            # turn gently towards target
            tw.angular.z = max(-self.angular_speed, min(self.angular_speed, yaw_err))
            if front_dist < self.min_clearance:
                # practically stopped
                tw.linear.x = 0.0
            self.cmd_vel_pub.publish(tw)
        else:
            # Move quickly
            tw = Twist()
            # If yaw error is large, turn in place
            if abs(yaw_err) > 0.3:
                tw.linear.x = 0.0
            else:
                tw.linear.x = self.fast_speed
            tw.angular.z = max(-self.angular_speed, min(self.angular_speed, yaw_err))
            self.cmd_vel_pub.publish(tw)

        return False

    def get_front_distance(self):
        """Check the LaserScan midpoint as a rough 'front' distance."""
        if not self.laser_ranges:
            return float('inf')
        mid = len(self.laser_ranges)//2
        return self.laser_ranges[mid]


###############################################################################
# 4. MAIN: LAUNCH BOTH DETECTOR + COVERAGE IN MULTITHREADED EXECUTOR
###############################################################################
def main(args=None):
    rclpy.init(args=args)

    # Provide a scenario name for logging, e.g. "hard" or "final_demo"
    scenario_name = "warehouse_improved"

    # 1) Detector node (could detect objects, do sensor fusion, log CSV).
    detector = AdvancedDetector(scenario_name=scenario_name)

    # 2) Coverage node (visits corners, random coverage, returns home).
    coverage = AdaptiveCoverage()

    # MultiThreadedExecutor so both spin concurrently
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    executor.add_node(coverage)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # When done, log final summary & shut down
        detector.finish_and_log_summary(note="User interrupted or run completed.")
        detector.destroy_node()
        coverage.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
