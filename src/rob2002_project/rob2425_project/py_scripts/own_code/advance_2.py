#!/usr/bin/env python3
"""
Full ROS2 script:
  1. AdvancedDetector node:
     - Subscribes to color + depth + LiDAR
     - Detects and logs colored objects (R/G/B) in CSV + terminal
  2. ZigZagReversibleCoverage node:
     - Plans a zigzag coverage over a bounding box
     - Avoids obstacles with brake/slow logic
     - Reverses + turns if too close
     - Accelerates up to max speed if clear
     - Returns home + stops at completion
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

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge


###############################################################################
# CSV LOGGER
###############################################################################
class CSVLogger:
    """
    Creates a CSV file to record DETECTION rows for each object
    and a SUMMARY row at the end.
    """
    def __init__(self, scenario_name="ZigZag_coverage"):
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"results_{scenario_name}_{now_str}.csv"

        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "record_type",  # DETECTION or SUMMARY
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
        t = time.time() - self.start_time
        self.writer.writerow([
            "DETECTION",
            f"{t:.2f}",
            color_label,
            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
            self.scenario_name,
            "", "", "", "",
            note
        ])
        self.file.flush()

    def log_summary(self, red_count, green_count, blue_count, note=""):
        t = time.time() - self.start_time
        total = red_count + green_count + blue_count
        self.writer.writerow([
            "SUMMARY",
            f"{t:.2f}",
            "",
            "", "", "",
            self.scenario_name,
            red_count,
            green_count,
            blue_count,
            total,
            note
        ])
        self.file.flush()

    def close(self):
        self.file.close()


###############################################################################
# ADVANCED DETECTOR NODE
###############################################################################
class AdvancedDetector(Node):
    """
    Detects R/G/B objects in an HSV color space, uses depth to find 3D coords,
    logs each detection in CSV and prints to terminal, avoids double counting.
    """

    def __init__(self, scenario_name="ZigZag_coverage"):
        super().__init__('advanced_detector')

        # CSV logger
        self.logger = CSVLogger(scenario_name=scenario_name)

        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'odom'      # or 'map'
        self.camera_frame = 'depth_link'

        # For camera models
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None

        # Data
        self.bridge = CvBridge()
        self.depth_image_ros = None
        self.real_robot = False

        # Store detected objects to avoid double counting
        self.detected_objects = []
        self.detection_threshold = 0.6

        # HSV thresholds for color detection
        # (In a real scenario, you'd calibrate or use a CNN).
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)

        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130,255,255],  dtype=np.uint8)

        # Subscribe to LiDAR as well (placeholder for sensor fusion).
        self.lidar_ranges = []
        self.create_subscription(
            LaserScan, 'scan',
            self.lidar_callback, qos.qos_profile_sensor_data)

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

        self.create_subscription(CameraInfo, ccamera_info_topic,
                                 self.ccamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(CameraInfo, dcamera_info_topic,
                                 self.dcamera_info_callback, qos.qos_profile_sensor_data)
        self.create_subscription(Image, cimage_topic,
                                 self.image_color_callback, qos.qos_profile_sensor_data)
        self.create_subscription(Image, dimage_topic,
                                 self.image_depth_callback, qos.qos_profile_sensor_data)

        self.visualise = True
        self.get_logger().info("AdvancedDetector initialized with CSV logging.")

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
        if self.ccamera_model and self.dcamera_model and (self.color2depth_aspect is None):
            c_aspect = (math.atan2(self.ccamera_model.width, 2*self.ccamera_model.fx())
                        / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2*self.dcamera_model.fx())
                        / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(f"color2depth_aspect = {self.color2depth_aspect:.3f}")

    def image_depth_callback(self, msg: Image):
        self.depth_image_ros = msg

    def image_color_callback(self, msg: Image):
        """
        Called ~30Hz or so. We detect color blobs, find their 3D location, and log new objects.
        """
        if (self.color2depth_aspect is None or self.depth_image_ros is None or
            not self.ccamera_model or not self.dcamera_model):
            return

        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.depth_image_ros, "32FC1")
        if self.real_robot:
            depth_img /= 1000.0  # convert mm->m

        # Convert to HSV
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # Red (two intervals)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

        # Green, Blue
        green_mask= cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask = cv2.inRange(hsv, self.lower_blue,  self.upper_blue)

        # Combine
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        contours, _   = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Avoid double counting in the same frame
        this_frame_bboxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:  # min area
                continue

            x, y, w, h = cv2.boundingRect(c)
            # check overlap with any existing box in this frame
            if any(self.rect_overlap(x, y, w, h, bx, by, bw, bh) for (bx,by,bw,bh) in this_frame_bboxes):
                continue
            this_frame_bboxes.append((x, y, w, h))

            # Determine color by summing each mask
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

            # Depth coordinate
            cx = x + w/2
            cy = y + h/2
            dx, dy = self.color2depth_coords(cx, cy, color_img, depth_img)
            if (dx < 0 or dy < 0 or
                dx >= depth_img.shape[1] or dy >= depth_img.shape[0]):
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

            # Check if new object (same color, within threshold => skip)
            if not self.is_new_object(gx, gy, gz, color_label):
                continue

            self.detected_objects.append({
                'color': color_label,
                'x': gx, 'y': gy, 'z': gz
            })

            # Log in CSV + print to terminal
            self.logger.log_detection(color_label, gx, gy, gz)
            self.get_logger().info(f"NEW {color_label} object at ({gx:.2f},{gy:.2f},{gz:.2f}). "
                                   f"Total so far: {len(self.detected_objects)}")

            # For debug display
            if self.visualise:
                cv2.rectangle(color_img, (x,y), (x+w,y+h), (255,255,0), 2)
                cv2.putText(color_img, color_label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

        # Visualization
        if self.visualise:
            cv2.imshow("Detector - Color", color_img)
            scaled_depth = depth_img / 5.0
            cv2.imshow("Detector - Depth", scaled_depth)
            cv2.waitKey(1)

    def rect_overlap(self, x1,y1,w1,h1, x2,y2,w2,h2):
        if x1 > x2+w2 or x2 > x1+w1:
            return False
        if y1 > y2+h2 or y2 > y1+h1:
            return False
        return True

    def color2depth_coords(self, cx, cy, color_img, depth_img):
        c_cx = color_img.shape[1]/2.0
        c_cy = color_img.shape[0]/2.0
        d_cx = depth_img.shape[1]/2.0
        d_cy = depth_img.shape[0]/2.0

        shift_x = cx - c_cx
        shift_y = cy - c_cy
        dx = d_cx + shift_x * self.color2depth_aspect
        dy = d_cy + shift_y * self.color2depth_aspect
        return dx, dy

    def is_new_object(self, gx, gy, gz, color_label):
        """
        If an existing object of the same color is within detection_threshold => not new.
        """
        for obj in self.detected_objects:
            if obj['color'] == color_label:
                dx = obj['x'] - gx
                dy = obj['y'] - gy
                dz = obj['z'] - gz
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < self.detection_threshold:
                    return False
        return True

    def finish_and_log_summary(self, note="Coverage done"):
        """
        Called at the end to log final summary of red/green/blue counts, then close CSV.
        """
        red_count   = sum(1 for o in self.detected_objects if o['color'] == 'red')
        green_count = sum(1 for o in self.detected_objects if o['color'] == 'green')
        blue_count  = sum(1 for o in self.detected_objects if o['color'] == 'blue')
        self.logger.log_summary(red_count, green_count, blue_count, note)
        self.logger.close()

        # Print final summary to terminal
        total = len(self.detected_objects)
        self.get_logger().info("=========================================")
        self.get_logger().info(f"Coverage Complete! Found {total} objects:")
        self.get_logger().info(f"  Red:   {red_count}")
        self.get_logger().info(f"  Green: {green_count}")
        self.get_logger().info(f"  Blue:  {blue_count}")
        idx=1
        for o in self.detected_objects:
            c = o['color']
            x = o['x']
            y = o['y']
            z = o['z']
            self.get_logger().info(f"{idx}. {c} at ({x:.2f}, {y:.2f}, {z:.2f})")
            idx+=1
        self.get_logger().info("=========================================")


###############################################################################
# HELPER: Build a zigzag path
###############################################################################
def create_zigzag_waypoints(x_min, x_max, y_min, y_max, x_step):
    """
    Creates a list of waypoints that covers the area in a "snake" pattern
    from x_min to x_max, stepping by x_step, going up or down in y at each pass.
    Then returns to x_min, y_min at the end.
    """
    waypoints = []
    cur_x = x_min
    going_up = True
    while cur_x <= x_max:
        if going_up:
            waypoints.append((cur_x, y_max))
        else:
            waypoints.append((cur_x, y_min))
        cur_x += x_step
        going_up = not going_up

    # We might also want to "snake" back
    # But a simpler approach is to finalize at the top-right:
    # then we do a direct line returning home at the end of coverage.
    # For completeness, let's end near bottom-left.
    waypoints.append((x_min, y_min))
    return waypoints


###############################################################################
# ZIGZAG COVERAGE NODE WITH ACCELERATION, OBSTACLE AVOIDANCE, REVERSE TURN
###############################################################################
class ZigZagReversibleCoverage(Node):
    """
    1. Generates zigzag waypoints to cover a bounding box.
    2. Accelerates if clear, brakes if obstacle encountered.
    3. If dangerously close, reverses and turns into the clear direction.
    4. Follows each waypoint in turn, then returns to start, then stops.
    """

    def __init__(self):
        super().__init__('zigzag_reversible_coverage')

        # Publishers/Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Laser data
        self.laser_ranges = []

        # Pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.home_x = None
        self.home_y = None

        # Coverage waypoints
        self.x_min = -4.5
        self.x_max =  4.5
        self.y_min = -4.5
        self.y_max =  4.5
        self.x_step= 1.0
        self.waypoints = create_zigzag_waypoints(
            self.x_min, self.x_max, self.y_min, self.y_max, self.x_step
        )
        self.wp_index = 0
        self.wp_tolerance = 0.4

        # Movement states
        self.done_coverage = False
        self.done = False

        # Reverse maneuver
        self.reversing = False
        self.reverse_timer = 0
        self.reverse_time_limit = 10  # ticks ~1 sec

        # Speed parameters
        self.max_speed = 0.5
        self.min_speed = 0.0
        self.current_speed = 0.0  # We'll accelerate up to max_speed
        self.accel_rate = 0.05    # m/s^2 each cycle
        self.decel_rate = 0.1

        self.angular_speed_limit = 0.4

        # Dist thresholds
        self.brake_distance = 1.0
        self.min_clearance  = 0.5

        self.get_logger().info("ZigZagReversibleCoverage node started. Covering warehouse...")

    # --------------------------------------------------------------------------
    # LASER + ODOM
    # --------------------------------------------------------------------------
    def laser_callback(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.x = p.x
        self.y = p.y
        o = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.yaw = yaw

        if self.home_x is None:
            self.home_x = self.x
            self.home_y = self.y
            self.get_logger().info(f"Home set at ({self.home_x:.2f}, {self.home_y:.2f})")

    # --------------------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------------------
    def timer_callback(self):
        if self.done:
            # Final stop
            self.cmd_vel_pub.publish(Twist())
            return

        if not self.laser_ranges:
            return

        # If reversing, handle that first
        if self.reversing:
            self.handle_reverse()
            return

        # If coverage not done, proceed with coverage
        if not self.done_coverage:
            self.coverage_step()
        else:
            # Return home
            self.go_home_step()

    # --------------------------------------------------------------------------
    # COVERAGE STEP: Follow waypoints in a zigzag pattern
    # --------------------------------------------------------------------------
    def coverage_step(self):
        if self.wp_index >= len(self.waypoints):
            self.done_coverage = True
            self.get_logger().info("All zigzag waypoints visited. Returning home.")
            return

        # Move to the next waypoint
        wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.x
        dy = wy - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < self.wp_tolerance:
            self.wp_index += 1
            return

        # Decide movement
        self.move_toward(wx, wy)

    # --------------------------------------------------------------------------
    # GO HOME STEP: After coverage, return to (home_x, home_y)
    # --------------------------------------------------------------------------
    def go_home_step(self):
        dx = self.home_x - self.x
        dy = self.home_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.7:
            self.get_logger().info("Arrived home. Coverage complete! Stopping now.")
            self.done = True
            return

        self.move_toward(self.home_x, self.home_y)

    # --------------------------------------------------------------------------
    # MOVE TOWARD (tx, ty)
    # --------------------------------------------------------------------------
    def move_toward(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_err = desired_yaw - self.yaw
        # Wrap to [-pi, pi]
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        front_dist = self.get_front_distance()
        twist = Twist()

        # If front_dist < min_clearance => REVERSE
        if front_dist < self.min_clearance:
            self.start_reverse()
            return

        # If front_dist < brake_distance => we decelerate or keep speed low
        if front_dist < self.brake_distance:
            # decelerate quickly
            self.current_speed = max(self.current_speed - self.decel_rate, 0.0)
            if self.current_speed < 0.05:
                self.current_speed = 0.05
        else:
            # accelerate up to max_speed
            self.current_speed = min(self.current_speed + self.accel_rate, self.max_speed)

        # If yaw error is large, reduce speed
        if abs(yaw_err) > 0.5:
            self.current_speed *= 0.5

        # Angular speed: basic P-control
        ang_speed = 1.0 * yaw_err
        # clamp
        ang_speed = max(-self.angular_speed_limit, min(self.angular_speed_limit, ang_speed))

        twist.linear.x = self.current_speed
        twist.angular.z = ang_speed
        self.cmd_vel_pub.publish(twist)

    # --------------------------------------------------------------------------
    # REVERSE MANEUVER
    # --------------------------------------------------------------------------
    def start_reverse(self):
        self.get_logger().warn("Too close to obstacle! Initiating reverse + turn.")
        self.reversing = True
        self.reverse_timer = 0
        # Immediately publish a zero speed to "brake"
        self.current_speed = 0.0
        tw = Twist()
        self.cmd_vel_pub.publish(tw)

    def handle_reverse(self):
        # Reverse + turn away from obstacle
        # Decide which side is clearer
        left_d = self.get_left_distance()
        right_d= self.get_right_distance()

        tw = Twist()
        tw.linear.x = -0.2  # reverse speed

        # Turn toward the bigger side
        if right_d > left_d:
            tw.angular.z = -0.4  # turn right
        else:
            tw.angular.z =  0.4  # turn left

        self.cmd_vel_pub.publish(tw)
        self.reverse_timer += 1
        if self.reverse_timer > self.reverse_time_limit:
            self.reversing = False
            self.reverse_timer = 0
            self.get_logger().info("Reverse maneuver complete. Resuming coverage.")

    # --------------------------------------------------------------------------
    # LASER UTILS
    # --------------------------------------------------------------------------
    def get_front_distance(self):
        if not self.laser_ranges:
            return float('inf')
        mid = len(self.laser_ranges)//2
        return self.laser_ranges[mid]

    def get_left_distance(self):
        if not self.laser_ranges:
            return float('inf')
        idx = len(self.laser_ranges)//2 + 90
        if idx >= len(self.laser_ranges):
            idx = len(self.laser_ranges)-1
        return self.laser_ranges[idx]

    def get_right_distance(self):
        if not self.laser_ranges:
            return float('inf')
        idx = len(self.laser_ranges)//2 - 90
        if idx < 0:
            idx = 0
        return self.laser_ranges[idx]


###############################################################################
# MAIN
###############################################################################
def main(args=None):
    rclpy.init(args=args)

    # 1) AdvancedDetector
    detector = AdvancedDetector(scenario_name="ZigZagFullCoverage")

    # 2) ZigZagReversibleCoverage
    coverage = ZigZagReversibleCoverage()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    executor.add_node(coverage)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # When shutting down, log final summary from the detector
        detector.finish_and_log_summary(note="Run completed or interrupted.")
        detector.destroy_node()
        coverage.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
