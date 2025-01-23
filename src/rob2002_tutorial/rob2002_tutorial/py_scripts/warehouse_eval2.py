#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.executors import MultiThreadedExecutor

# ROS messages
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (
    Pose, PoseStamped, Point, Quaternion,
    PoseArray
)

# TF / transforms
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import image_geometry

# For bridging ROS <-> OpenCV
from cv_bridge import CvBridge

# Nav2 imports
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

###############################################################################
# 1. DETECTOR3D (from professor's Detector3D)
###############################################################################
class Detector3D(Node):
    """
    Subscribes to color + depth images, detects big color blobs (BGR),
    transforms them into global frame, and publishes each detection
    to '/object_location' as a PoseStamped.
    """

    def __init__(self):
        super().__init__('detector_3d')

        # If using real robot, set to True to scale depth from mm -> m, etc.
        self.real_robot = False

        # Camera models for color + depth
        self.ccamera_model = None
        self.dcamera_model = None

        # Depth image storage
        self.image_depth_ros = None
        self.color2depth_aspect = None

        self.min_area_size = 100
        self.global_frame = 'odom'   # or 'map' if you have a map
        self.camera_frame = 'depth_link'

        # For optional debug windows
        self.visualisation = True
        self.bridge = CvBridge()

        # TF buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # If your topics differ, update them here
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
        self.create_subscription(CameraInfo, ccamera_info_topic,
                                 self.ccamera_info_callback,
                                 qos.qos_profile_sensor_data)
        self.create_subscription(CameraInfo, dcamera_info_topic,
                                 self.dcamera_info_callback,
                                 qos.qos_profile_sensor_data)
        self.create_subscription(Image, cimage_topic,
                                 self.image_color_callback,
                                 qos.qos_profile_sensor_data)
        self.create_subscription(Image, dimage_topic,
                                 self.image_depth_callback,
                                 qos.qos_profile_sensor_data)

        # Publisher: each found object's location as PoseStamped
        self.object_location_pub = self.create_publisher(
            PoseStamped, '/object_location', qos.qos_profile_parameters
        )

        self.get_logger().info("Detector3D node started.")

    def ccamera_info_callback(self, data):
        if self.ccamera_model is None:
            self.ccamera_model = image_geometry.PinholeCameraModel()
            self.ccamera_model.fromCameraInfo(data)
            self.update_color2depth_aspect()

    def dcamera_info_callback(self, data):
        if self.dcamera_model is None:
            self.dcamera_model = image_geometry.PinholeCameraModel()
            self.dcamera_model.fromCameraInfo(data)
            self.update_color2depth_aspect()

    def update_color2depth_aspect(self):
        # approximate ratio used to map color pixel coords -> depth coords
        if (self.color2depth_aspect is None and
                self.ccamera_model and self.dcamera_model):
            # horizontal FOV color:
            c_aspect = (math.atan2(self.ccamera_model.width,
                                   2 * self.ccamera_model.fx())
                        / self.ccamera_model.width)
            # horizontal FOV depth:
            d_aspect = (math.atan2(self.dcamera_model.width,
                                   2 * self.dcamera_model.fx())
                        / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(
                f"color2depth_aspect = {self.color2depth_aspect:.3f}"
            )

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_color_callback(self, data):
        # only process if we have camera models + at least one depth frame
        if (self.color2depth_aspect is None or
                self.image_depth_ros is None or
                self.ccamera_model is None or
                self.dcamera_model is None):
            return

        # Convert color + depth to OpenCV
        image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        if self.real_robot:
            # if real robot publishes depth in mm, convert
            image_depth /= 1000.0

        # We'll detect red, green, and blue in BGR:
        red_mask   = cv2.inRange(image_color, (0,   0,   80), (50,  50,  255))
        green_mask = cv2.inRange(image_color, (0,   80,  0 ), (50,  255, 50 ))
        blue_mask  = cv2.inRange(image_color, (80,  0,   0 ), (255, 50,  50 ))
        combined   = cv2.bitwise_or(red_mask,
                     cv2.bitwise_or(green_mask, blue_mask))

        # find contours
        contours, _ = cv2.findContours(combined,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # For each large contour, compute centroid in image, transform to global
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_area_size:
                continue

            # Centroid from image moments
            moments = cv2.moments(cnt)
            if moments["m00"] == 0.0:
                continue
            cy = moments["m01"] / moments["m00"]
            cx = moments["m10"] / moments["m00"]

            # map from color coords -> approximate depth coords
            dx, dy = self.color2depth_coords(cx, cy, image_color, image_depth)
            if (dx < 0 or dy < 0 or
                    dx >= image_depth.shape[1] or
                    dy >= image_depth.shape[0]):
                continue

            depth_val = image_depth[int(dy), int(dx)]
            if depth_val <= 0.0 or math.isnan(depth_val) or math.isinf(depth_val):
                continue

            # project into camera coordinates
            camera_pose = self.pixel_to_3d_camera(cx, cy, depth_val)

            # transform camera -> global
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame, self.camera_frame,
                    rclpy.time.Time())
                global_pose = do_transform_pose(camera_pose, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            # Publish as PoseStamped
            ps = PoseStamped(
                header=Header(frame_id=self.global_frame),
                pose=global_pose
            )
            self.object_location_pub.publish(ps)

            # Debug prints
            self.get_logger().info(
                f"[Detector3D] Found object {idx} at global: "
                f"({global_pose.position.x:.2f}, "
                f"{global_pose.position.y:.2f}, "
                f"{global_pose.position.z:.2f})"
            )

            # optional: draw circle in color image
            if self.visualisation:
                cv2.circle(image_color, (int(cx), int(cy)), 5, (255,255,0), -1)

        # show debug windows
        if self.visualisation:
            show_color = image_color.copy()
            show_depth = image_depth.copy() / 10.0  # scale for visibility
            # optionally resize to half
            show_color = cv2.resize(show_color, None, fx=0.5, fy=0.5)
            show_depth = cv2.resize(show_depth, None, fx=0.5, fy=0.5)
            cv2.imshow("Detector3D - color", show_color)
            cv2.imshow("Detector3D - depth", show_depth)
            cv2.waitKey(1)

    def color2depth_coords(self, cx, cy, color_img, depth_img):
        """Approx shift from color pixel coords -> depth coords."""
        c_cx = color_img.shape[1] / 2.0
        c_cy = color_img.shape[0] / 2.0
        d_cx = depth_img.shape[1] / 2.0
        d_cy = depth_img.shape[0] / 2.0

        shift_x = cx - c_cx
        shift_y = cy - c_cy
        dx = d_cx + shift_x * self.color2depth_aspect
        dy = d_cy + shift_y * self.color2depth_aspect
        return dx, dy

    def pixel_to_3d_camera(self, cx, cy, depth_val):
        """
        Convert a pixel (cx,cy) at 'depth_val' into a Pose
        in the camera coordinate system, using the camera model.
        """
        # projectPixelTo3dRay wants (u, v) in (column, row) => (cx, cy)
        ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
        # scale by depth so that z in camera coords == depth_val
        ray *= depth_val / ray[2]

        pose = Pose()
        pose.position.x = float(ray[0])
        pose.position.y = float(ray[1])
        pose.position.z = float(ray[2])
        pose.orientation.w = 1.0
        return pose


###############################################################################
# 2. COUNTER3D (from professor's counter_3d.py)
###############################################################################
class Counter3D(Node):
    """
    Subscribes to '/object_location' (PoseStamped),
    checks if each new detection is > detection_threshold from previous ones,
    if so, adds to the list of unique objects.

    Publishes PoseArray of all object poses to '/object_count_array'
    for RViZ visualisation. Prints total count in the console.
    """

    def __init__(self):
        super().__init__('counter_3d')

        # how close in 3D to consider "same" object
        self.detection_threshold = 0.2

        # storage of unique object poses
        self.detected_objects = []

        # subscribe to object detector
        self.create_subscription(
            PoseStamped, '/object_location',
            self.counter_callback,
            qos.qos_profile_sensor_data
        )

        # publish all detected objects as an array of poses
        self.publisher = self.create_publisher(
            PoseArray, '/object_count_array',
            qos.qos_profile_default
        )

        self.get_logger().info("Counter3D node started.")

    def counter_callback(self, msg: PoseStamped):
        new_pose = msg.pose
        object_exists = False

        # Check if new_pose is within detection_threshold of any stored
        for existing_pose in self.detected_objects:
            dx = existing_pose.position.x - new_pose.position.x
            dy = existing_pose.position.y - new_pose.position.y
            dz = existing_pose.position.z - new_pose.position.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < self.detection_threshold:
                object_exists = True
                break

        if not object_exists:
            self.detected_objects.append(new_pose)
            self.get_logger().info(
                f"[Counter3D] New object! total = {len(self.detected_objects)}"
            )

        # publish as PoseArray
        parray = PoseArray()
        parray.header = msg.header  # same frame, timestamp as the detection
        parray.poses = self.detected_objects
        self.publisher.publish(parray)


###############################################################################
# 3. NAVIGATOR NODE: BasicNavigator for waypoints
###############################################################################
class NavigatorNode(Node):
    """
    Creates a Nav2 BasicNavigator, sets some sample waypoints,
    and drives the robot around. You can adapt these coordinates
    to your warehouse / map as needed.
    """

    def __init__(self):
        super().__init__('warehouse_navigator')

        self.navigator = BasicNavigator()
        self.declare_parameter('use_initial_pose', True)

        # Example list of waypoints in (x, y, theta) with respect to 'map' or 'odom'
        self.waypoints = [
            (1.0,  0.0,  0.0),
            (1.0, -1.0, -1.57),
            (0.0, -1.0,  3.14),
            (0.0,  0.0,  1.57),
        ]

        # Start a timer to run the navigation logic once
        self.init_timer = self.create_timer(1.0, self.start_navigation)
        self.is_running = False

    def start_navigation(self):
        if self.is_running:
            return

        self.is_running = True
        self.init_timer.cancel()

        self.get_logger().info("NavigatorNode: Starting Nav2 BasicNavigator...")

        # Set initial pose (if needed)
        use_initial_pose = self.get_parameter('use_initial_pose').value
        if use_initial_pose:
            init_pose = PoseStamped()
            init_pose.header.frame_id = 'map'
            init_pose.header.stamp = self.navigator.get_clock().now().to_msg()

            # Example: place the robot near (0,0)
            init_pose.pose.position.x = 0.0
            init_pose.pose.position.y = 0.0
            init_pose.pose.orientation.z = 0.0
            init_pose.pose.orientation.w = 1.0

            self.navigator.setInitialPose(init_pose)

        # Wait until nav2 is active
        self.navigator.waitUntilNav2Active()

        # Build a list of PoseStamped for each waypoint
        pose_array = []
        for (x, y, th) in self.waypoints:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = self.navigator.get_clock().now().to_msg()
            # Convert Euler to quaternion
            q = quaternion_from_euler(0, 0, th)
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.z = q[2]
            ps.pose.orientation.w = q[3]
            pose_array.append(ps)

        self.get_logger().info("Sending robot through waypoints...")
        self.navigator.followWaypoints(pose_array)

        # Create a timer to check for completion
        self.feedback_timer = self.create_timer(0.5, self.check_feedback)

    def check_feedback(self):
        if not self.navigator.isTaskComplete():
            # just show periodic feedback
            feedback = self.navigator.getFeedback()
            if feedback:
                idx = feedback.current_waypoint
                self.get_logger().info(
                    f"Currently traveling to waypoint {idx+1}/{len(self.waypoints)}"
                )
            return

        # If task complete, see result
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Waypoints complete successfully!")
        elif result == TaskResult.FAILED:
            self.get_logger().error("Navigation task failed!")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Navigation task was canceled!")
        else:
            self.get_logger().info("Unknown result code. Possibly incomplete?")

        # Done. Cancel any velocity or tasks
        self.navigator.cancelTask()
        self.feedback_timer.cancel()
        self.get_logger().info("NavigatorNode: Finished route.")


###############################################################################
# 4. MAIN: Spin Everything in MultiThreadedExecutor
###############################################################################
def main(args=None):
    rclpy.init(args=args)

    # Create the three nodes
    detector_node = Detector3D()
    counter_node  = Counter3D()
    navigator_node= NavigatorNode()

    # Use a multi-threaded executor so they run simultaneously
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(detector_node)
    executor.add_node(counter_node)
    executor.add_node(navigator_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detector_node.destroy_node()
        counter_node.destroy_node()
        navigator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
