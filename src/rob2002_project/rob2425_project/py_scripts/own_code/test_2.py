#!/usr/bin/env python3
"""
Robot Evaluation Script with:
  - Advanced Detector (unchanged from prior example).
  - AdaptiveCoverage with improved obstacle avoidance:
      • The robot will not come closer than its own length (or safe_distance) to any obstacle.
      • When an obstacle is detected, it chooses a single direction (left or right) based on available clear space and the current target's direction.
      • That avoidance decision is "locked in" until the obstacle is cleared.
      • Once clear, the robot resumes navigation toward its waypoint.
  - All events are relayed to the terminal and logged in CSV.
  - The robot completes its task and stops when finished.
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
# 1. CSV LOGGER – For Recording Results (same as before)
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

    def log_detection(self, record_type, x, y, z, note=""):
        t_elapsed = time.time() - self.start_time
        self.writer.writerow([
            record_type,  # e.g. "red", "obstacle", "clear"
            f"{t_elapsed:.2f}",
            "",
            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
            self.scenario_name,
            "", "", "", "",
            note
        ])
        self.file.flush()

    def log_summary(self, count_red, count_green, count_blue, note="End of run"):
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
            blue_count if (blue_count := count_blue) else count_blue,
            total,
            note
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

        # CSV Logger
        self.logger = CSVLogger(scenario_name=scenario_name)

        # TF / frames
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

        # Storage for unique detections
        self.detected_objects = []
        self.detection_threshold = 0.6

        # Example HSV thresholds for color
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)

        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)

        # LiDAR placeholder
        self.lidar_ranges = []
        self.create_subscription(LaserScan,'scan',self.lidar_callback,qos.qos_profile_sensor_data)

        # Camera info topics
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

        self.create_subscription(CameraInfo,ccamera_info_topic,self.ccamera_info_callback,qos.qos_profile_sensor_data)
        self.create_subscription(CameraInfo,dcamera_info_topic,self.dcamera_info_callback,qos.qos_profile_sensor_data)
        self.create_subscription(Image,cimage_topic,self.image_color_callback,qos.qos_profile_sensor_data)
        self.create_subscription(Image,dimage_topic,self.image_depth_callback,qos.qos_profile_sensor_data)

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
        green_mask= cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask = cv2.inRange(hsv, self.lower_blue,  self.upper_blue)

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        contours, _   = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_bboxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # Skip overlapping bboxes.
            if any(self.overlaps(x, y, w, h, px, py, pw, ph) for (px,py,pw,ph) in frame_bboxes):
                continue
            frame_bboxes.append((x, y, w, h))

            # Determine color label.
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

            # Project to 3D using pinhole camera model.
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

            # Skip if object already counted.
            if not self.is_new_object(gx, gy, gz, color_label):
                continue

            self.detected_objects.append({'color': color_label, 'x': gx, 'y': gy, 'z': gz})
            self.logger.log_detection(color_label, gx, gy, gz)

            # Visualization.
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
        # Check if two bounding boxes overlap.
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
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < self.detection_threshold:
                    return False
        return True

    def finish_and_log_summary(self, note="End of run"):
        red_count   = sum(1 for o in self.detected_objects if o['color'] == 'red')
        green_count = sum(1 for o in self.detected_objects if o['color'] == 'green')
        blue_count  = sum(1 for o in self.detected_objects if o['color'] == 'blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()


###############################################################################
# 3. ADAPTIVE COVERAGE with IMPROVED OBSTACLE AVOIDANCE
###############################################################################
class AdaptiveCoverage(Node):
    """
    - Visits pre-defined corners of the warehouse and then performs random coverage.
    - At all times the robot checks its laser-scan.
    - If any obstacle (wall or colored box) is detected closer than safe_distance,
      the robot overrides its current drive command.
      An avoidance direction (either left or right) is chosen only once per avoidance event,
      using both available clear space and the direction toward the current target.
    - Once a clear path is detected, the robot resets the avoidance state and resumes its task.
    - The robot returns home and stops when all planned targets are visited.
    """
    def __init__(self):
        super().__init__('adaptive_coverage')

        # Movement topics.
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_cb, 10)

        # Timer for periodic control updates.
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Laser & Pose data.
        self.laser_ranges = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Task state.
        self.state = "VISIT_CORNERS"
        self.corner_idx = 0
        self.done = False

        # Warehouse corners.
        self.corners = [
            ( 4.5,  4.5),
            ( 4.5, -4.5),
            (-4.5, -4.5),
            (-4.5,  4.5),
        ]
        self.corner_tolerance = 0.7

        # Extra random coverage targets.
        self.random_targets = []
        self.rng = random.Random()

        # Start position for returning home.
        self.start_x = None
        self.start_y = None

        # Speed parameters.
        self.fast_speed = 0.5
        self.slow_speed = 0.15
        self.angular_speed = 0.3

        # Distance thresholds.
        # safe_distance is the closest allowed distance from any object.
        self.safe_distance = 0.5   # adjust as needed (roughly the robot’s length)
        self.brake_distance = 1.0  # if obstacles are farther than this, robot can accelerate

        # Variables to store avoidance state.
        self.avoidance_active = False
        self.avoidance_direction = 0.0  # +ve for left, -ve for right

        # CSV logger for obstacle events.
        self.obstacle_logger = CSVLogger(scenario_name="AdaptiveCoverage_Obstacles")
        self.get_logger().info("AdaptiveCoverage node ready (with improved obstacle avoidance).")

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
            self.get_logger().info(f"Start position set: ({px:.2f}, {py:.2f})")

    def get_current_target(self):
        """
        Returns the current waypoint target as an (x, y) tuple based on the current state.
        """
        if self.state == "VISIT_CORNERS" and self.corner_idx < len(self.corners):
            return self.corners[self.corner_idx]
        elif self.state == "RANDOM_COVERAGE" and self.random_targets:
            return self.random_targets[0]
        elif self.state == "RETURN_HOME":
            return (self.start_x, self.start_y)
        else:
            return None

    def timer_cb(self):
        if self.done:
            self.cmd_vel_pub.publish(Twist())
            return

        if not self.laser_ranges:
            return

        # Determine the current target from the state.
        current_target = self.get_current_target()

        # First, check whether obstacle avoidance is needed.
        if self.avoid_obstacle(current_target):
            return  # avoidance command has been published

        # If no avoidance is needed, reset avoidance state.
        self.avoidance_active = False
        self.avoidance_direction = 0.0

        # Proceed with planned navigation.
        if self.state == "VISIT_CORNERS":
            if self.corner_idx < len(self.corners):
                target = self.corners[self.corner_idx]
                reached = self.navigate_to_target(target)
                if reached:
                    self.get_logger().info(f"Corner {self.corner_idx+1} reached.")
                    self.corner_idx += 1
            else:
                self.get_logger().info("Corners done. Adding random coverage points.")
                for _ in range(5):
                    rx = self.rng.uniform(-4.5, 4.5)
                    ry = self.rng.uniform(-4.5, 4.5)
                    self.random_targets.append((rx, ry))
                self.state = "RANDOM_COVERAGE"

        elif self.state == "RANDOM_COVERAGE":
            if self.random_targets:
                tgt = self.random_targets[0]
                reached = self.navigate_to_target(tgt)
                if reached:
                    self.random_targets.pop(0)
            else:
                self.get_logger().info("Random coverage done. Returning home.")
                self.state = "RETURN_HOME"

        elif self.state == "RETURN_HOME":
            home = (self.start_x, self.start_y)
            reached = self.navigate_to_target(home)
            if reached:
                self.get_logger().info("Reached home. DONE.")
                self.done = True

    def avoid_obstacle(self, target):
        """
        Checks the entire laser-scan for objects closer than safe_distance.
        If one is found, and if not already in avoidance mode, a single avoidance direction
        is chosen based on (a) which side (left or right) has better clearance and (b) the direction
        toward the target.
        While an obstacle is detected, the robot commands a fixed turning rate.
        Returns True if avoidance was executed, otherwise False.
        """
        if not self.laser_ranges:
            return False

        min_range = min(self.laser_ranges)
        if min_range < self.safe_distance:
            # If not already in avoidance mode, decide on one direction.
            if not self.avoidance_active:
                total_scans = len(self.laser_ranges)
                mid = total_scans // 2
                left_avg = np.mean(self.laser_ranges[mid:]) if total_scans - mid > 0 else 0
                right_avg = np.mean(self.laser_ranges[:mid]) if mid > 0 else 0

                # Set a default decision based on available clear space.
                if left_avg > right_avg:
                    decision_clear = "left"
                else:
                    decision_clear = "right"

                # If a target is provided, compute desired heading toward that target.
                if target is not None:
                    tx, ty = target
                    desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
                    yaw_error = desired_yaw - self.current_yaw
                    yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                    # If the target lies to the left, then prefer left; else right.
                    decision_target = "left" if yaw_error > 0 else "right"
                else:
                    decision_target = decision_clear

                # Choose the final decision.
                # Here you could weight one decision over the other.
                if decision_clear == decision_target:
                    final_decision = decision_clear
                else:
                    # If conflicting, choose the side with higher clearance.
                    final_decision = decision_clear

                self.avoidance_active = True
                if final_decision == "left":
                    self.avoidance_direction = self.angular_speed
                else:
                    self.avoidance_direction = -self.angular_speed

                self.get_logger().warn(f"ALERT: Obstacle detected at {min_range:.2f} m! Steering {final_decision}.")
                self.obstacle_logger.log_detection("obstacle", self.current_x, self.current_y, 0,
                                                   note=f"Obstacle at {min_range:.2f} m, steering {final_decision}")

            # Publish the fixed avoidance command.
            tw = Twist()
            tw.linear.x = 0.0  # remain in place while avoiding
            tw.angular.z = self.avoidance_direction
            self.cmd_vel_pub.publish(tw)
            return True
        return False

    def navigate_to_target(self, target):
        """
        Navigates toward the given target (x, y).
        Returns True if the target is reached.
        """
        tx, ty = target
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)
        if dist < self.corner_tolerance:
            self.cmd_vel_pub.publish(Twist())
            return True

        desired_yaw = math.atan2(dy, dx)
        yaw_err = desired_yaw - self.current_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        tw = Twist()
        # Check a narrow front sector.
        total_scans = len(self.laser_ranges)
        mid = total_scans // 2
        idx_start = max(0, mid - 5)
        idx_end = min(total_scans, mid + 5)
        front_sector = self.laser_ranges[idx_start:idx_end]
        front_min = min(front_sector) if front_sector else float('inf')

        if front_min > self.brake_distance:
            tw.linear.x = self.fast_speed
        else:
            tw.linear.x = self.slow_speed

        tw.angular.z = max(-self.angular_speed, min(self.angular_speed, yaw_err))
        self.cmd_vel_pub.publish(tw)
        return False


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
