#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An integrated ROS 2 script that:
 1) Launches a camera-based color detection node with HSV thresholds,
 2) Logs detections and final summary to CSV,
 3) Uses Nav2's BasicNavigator to follow a series of map-based waypoints,
 4) Prints progress and detection info to the console,
 5) Demonstrates robust usage for an academic / PhD-level demonstration.

Dependencies:
  - ROS 2 (Galactic or newer) with nav2 stack, image_pipeline, etc.
  - nav2_simple_commander (pip install or from your ROS2 workspace).
  - OpenCV + cv_bridge + image_geometry
  - tf_transformations
  - Ensure a valid TF tree with 'odom' or 'map' frame, plus camera frames.
  - A properly set up map server (with your "my_map.yaml"), AMCL, etc.
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

# ROS messages
from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

# TF, transforms, etc.
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

# For bridging ROS Image <-> OpenCV
from cv_bridge import CvBridge

# For navigation
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

###############################################################################
# 1. CSV LOGGER – For Recording Results
###############################################################################
class CSVLogger:
    """
    Records all color detections and final summary into a CSV for analysis.
    """
    def __init__(self, scenario_name="warehouse_evaluation"):
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
    """
    Subscribes to RGB + depth camera topics and LiDAR scans (optional).
    Detects colored boxes using HSV thresholds (red/green/blue).
    Avoids double-counting objects via a 3D position proximity check.
    Logs detections to CSV in real time.
    """
    def __init__(self, scenario_name="warehouse_evaluation"):
        super().__init__('advanced_detector')

        # CSV Logger
        self.logger = CSVLogger(scenario_name=scenario_name)

        # TF / frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'map'        # or 'odom', if you prefer
        self.camera_frame = 'camera_link'  # adjust to match your camera frame

        # Camera models (for depth alignment)
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None

        self.bridge = CvBridge()
        self.depth_image_ros = None
        self.real_robot = False  # set True if using real camera topics (scaling depth?)

        # Storage for unique detections
        self.detected_objects = []
        self.detection_threshold = 0.6  # how close is "the same object"

        # HSV thresholds for color boxes (tune as needed)
        # Red is split into two segments in HSV
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)

        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)

        # Example LiDAR subscription, if you want
        self.lidar_ranges = []
        self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos.qos_profile_sensor_data
        )

        # Camera info topics (update to your actual camera topics)
        ccamera_info_topic = '/limo/depth_camera_link/camera_info'
        dcamera_info_topic = '/limo/depth_camera_link/depth/camera_info'
        cimage_topic       = '/limo/depth_camera_link/image_raw'
        dimage_topic       = '/limo/depth_camera_link/depth/image_raw'

        # If real robot, sometimes these topics differ:
        if self.real_robot:
            ccamera_info_topic = '/camera/color/camera_info'
            dcamera_info_topic = '/camera/depth/camera_info'
            cimage_topic       = '/camera/color/image_raw'
            dimage_topic       = '/camera/depth/image_raw'
            self.camera_frame  = 'camera_color_optical_frame'

        # Subscriptions
        self.create_subscription(
            CameraInfo,
            ccamera_info_topic,
            self.ccamera_info_callback,
            qos.qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            dcamera_info_topic,
            self.dcamera_info_callback,
            qos.qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            cimage_topic,
            self.image_color_callback,
            qos.qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            dimage_topic,
            self.image_depth_callback,
            qos.qos_profile_sensor_data
        )

        # Visualization in local window (set False for headless)
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
        """
        Estimate how to shift from color pixels to depth pixels
        if color and depth resolutions differ.
        """
        if (self.ccamera_model and self.dcamera_model and self.color2depth_aspect is None):
            c_aspect = (math.atan2(self.ccamera_model.width, 2*self.ccamera_model.fx()) /
                        self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2*self.dcamera_model.fx()) /
                        self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(
                f"color2depth_aspect = {self.color2depth_aspect:.3f}"
            )

    def image_depth_callback(self, msg: Image):
        self.depth_image_ros = msg

    def image_color_callback(self, msg: Image):
        """
        Main color detection callback.
        1) Convert color + depth to OpenCV,
        2) Find color contours in HSV,
        3) Compute 3D position in global (map/odom) frame,
        4) Log new detections to CSV,
        5) Optional live display with bounding boxes.
        """
        if (self.color2depth_aspect is None or
            self.depth_image_ros is None or
            not self.ccamera_model or
            not self.dcamera_model):
            return

        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.depth_image_ros, "32FC1")
        if self.real_robot:
            depth_img /= 1000.0  # convert mm->m if needed

        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # Threshold by color
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask= cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask = cv2.inRange(hsv, self.lower_blue,  self.upper_blue)

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_bboxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:  # skip small noise
                continue

            x, y, w, h = cv2.boundingRect(c)
            # skip if overlapping an existing rect
            if any(self.overlaps(x, y, w, h, px, py, pw, ph) for (px,py,pw,ph) in frame_bboxes):
                continue
            frame_bboxes.append((x,y,w,h))

            # Determine color label in that region-of-interest
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

            # Find center of bounding box in color image
            cx = x + w/2.0
            cy = y + h/2.0

            # Convert to depth image coords
            dx, dy = self.color2depth_coords(cx, cy, color_img, depth_img)
            if (dx<0 or dy<0 or dx>=depth_img.shape[1] or dy>=depth_img.shape[0]):
                continue

            zval = depth_img[int(dy), int(dx)]
            if zval<=0.0 or math.isnan(zval) or math.isinf(zval):
                continue

            # Project pixel -> camera 3D
            from image_geometry import PinholeCameraModel
            ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
            # Scale by depth
            ray *= (zval / ray[2])

            camera_pose = Pose()
            camera_pose.position.x = ray[0]
            camera_pose.position.y = ray[1]
            camera_pose.position.z = ray[2]
            camera_pose.orientation.w = 1.0

            # Transform camera frame -> global frame
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

            # Avoid double counting the same object
            if not self.is_new_object(gx, gy, gz, color_label):
                continue

            # Record detection
            self.detected_objects.append({'color':color_label,'x':gx,'y':gy,'z':gz})
            self.logger.log_detection(color_label, gx, gy, gz)

            # (Optional) Visual
            if self.visualise:
                cv2.rectangle(color_img, (x,y), (x+w,y+h), (255,255,0), 2)
                cv2.putText(color_img, color_label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

        if self.visualise:
            cv2.imshow("AdvancedDetector - Color", color_img)
            scaled_depth = depth_img / 5.0  # just for better visualization
            cv2.imshow("AdvancedDetector - Depth", scaled_depth)
            cv2.waitKey(1)

    def overlaps(self, x1, y1, w1, h1, x2, y2, w2, h2):
        # bounding box overlap check
        if (x1 > x2+w2 or x2 > x1+w1 or y1 > y2+h2 or y2 > y1+h1):
            return False
        return True

    def color2depth_coords(self, cx, cy, color_img, depth_img):
        """
        Convert color pixel coords -> depth pixel coords 
        using an approximate scaling factor from update_color2depth_aspect().
        """
        c_cx = color_img.shape[1]/2.0
        c_cy = color_img.shape[0]/2.0
        d_cx = depth_img.shape[1]/2.0
        d_cy = depth_img.shape[0]/2.0

        shift_x = cx - c_cx
        shift_y = cy - c_cy

        dx = d_cx + shift_x*self.color2depth_aspect
        dy = d_cy + shift_y*self.color2depth_aspect
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
        red_count   = sum(1 for o in self.detected_objects if o['color']=='red')
        green_count = sum(1 for o in self.detected_objects if o['color']=='green')
        blue_count  = sum(1 for o in self.detected_objects if o['color']=='blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()


###############################################################################
# 3. WAYPOINT NAVIGATION NODE (using Nav2)
###############################################################################
class WarehouseNavigator(Node):
    """
    A node that:
      1) Creates a Nav2 'BasicNavigator',
      2) Waits for activation,
      3) Sends the robot through a series of waypoints in the warehouse,
      4) Publishes 'current_waypoint' for RViZ visualization,
      5) Prints feedback to console for demonstration.

    Let Nav2 handle obstacle avoidance, local/global costmaps, etc.
    """
    def __init__(self):
        super().__init__('warehouse_navigator')

        self.publisher_wp = self.create_publisher(
            PoseStamped, '/current_waypoint', qos.qos_profile_system_default
        )

        # We will start our BasicNavigator in a moment
        self.navigator = None
        self.timer = self.create_timer(1.0, self.init_and_run_once)
        self.executing = False
        self.done = False

        # Example waypoints covering corners / areas of interest:
        self.waypoint_route = [
            [ 4.5,  4.5,  math.pi],      # top-right corner
            [ 4.5, -4.5, -math.pi/2],    # bottom-right corner
            [-4.5, -4.5,  0.0],          # bottom-left corner
            [-4.5,  4.5,  math.pi/2],    # top-left corner
            [ 0.0,   0.0,  0.0],         # maybe center
        ]
        # Feel free to tweak these or load them from file

    def init_and_run_once(self):
        """
        One-time setup of the Nav2 BasicNavigator, then run the waypoints.
        This approach avoids doing it in __init__ (which can cause timing issues).
        """
        if self.done or self.executing:
            return

        self.get_logger().info("Initializing BasicNavigator and starting route...")

        # Create BasicNavigator
        self.navigator = BasicNavigator()

        # If you have an initial pose, set it here. Example:
        # (If not set, Nav2 will estimate from AMCL or initial pose)
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.orientation.z = 0.0
        initial_pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(initial_pose)

        # Wait for Nav2 to activate
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 is active. Following warehouse waypoints...")

        # Build a list of PoseStamped for each waypoint
        waypoints = []
        for wp in self.waypoint_route:
            x, y, th = wp
            pose_stamped = self.pose_from_xytheta(x, y, th)
            waypoints.append(pose_stamped)

        # Start the plan
        self.navigator.followWaypoints(waypoints)
        self.executing = True
        self.timer.callback = self.feedback_check  # change the timer callback

    def feedback_check(self):
        """
        Periodically check if the Nav2 waypoint-following is done,
        and publish the current waypoint for RViZ display.
        """
        if not self.executing:
            return

        # If we're not done, get feedback and keep publishing
        if not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                current_waypoint_idx = feedback.current_waypoint
                # Safety check
                if current_waypoint_idx < len(self.waypoint_route):
                    cur_pose = self.pose_from_xytheta(
                        *self.waypoint_route[current_waypoint_idx]
                    )
                    self.publisher_wp.publish(cur_pose)
                    self.get_logger().info(
                        f"Traversing waypoint {current_waypoint_idx+1}/"
                        f"{len(self.waypoint_route)} ..."
                    )
            return

        # If the task is complete, log the result
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("All waypoints succeeded!")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Navigation task was canceled!")
        elif result == TaskResult.FAILED:
            self.get_logger().error("Navigation task failed!")
        else:
            self.get_logger().info("Unknown result code.")

        self.done = True
        self.executing = False

        # Stop the robot's velocity
        self.navigator.cancelTask()
        self.get_logger().info("Route finished. Node is done.")

    def pose_from_xytheta(self, x, y, theta):
        """
        Helper to build PoseStamped from (x, y, theta).
        """
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        q = quaternion_from_euler(0, 0, theta)
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose


###############################################################################
# 4. MAIN: Launch Detector + Navigator in Multi-Threaded Executor
###############################################################################
def main(args=None):
    rclpy.init(args=args)

    scenario_name = "warehouse_coverage"

    # Instantiate our two main nodes
    detector_node = AdvancedDetector(scenario_name=scenario_name)
    nav_node = WarehouseNavigator()

    # Spin them in a multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector_node)
    executor.add_node(nav_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Log final detection summary
        detector_node.finish_and_log_summary(note="Run completed or interrupted.")

        # Cleanup
        detector_node.destroy_node()
        nav_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
