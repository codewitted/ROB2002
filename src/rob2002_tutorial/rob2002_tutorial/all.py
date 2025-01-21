#!/usr/bin/env python3
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy import qos

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge


###############################################################################
# Helpers: 2D bounding-box for per-frame detection, plus zigzag waypoint generator
###############################################################################
class Rectangle2D:
    """Simple bounding box in image coordinates to avoid repeated detection within one frame."""
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def overlaps(self, other) -> bool:
        """Return True if this rectangle overlaps with 'other' in 2D pixel space."""
        if (self.x1 > other.x2 or other.x1 > self.x2 or
            self.y1 > other.y2 or other.y1 > self.y2):
            return False
        return True

def create_zigzag_waypoints(x_min, x_max, y_min, y_max, x_step):
    """
    Build a list of (x,y) waypoints that zigzag vertically between y_min and y_max,
    stepping in X after each pass, then come back across. 
    You can adjust or remove the 'return' portion if you only want a single pass.
    """
    waypoints = []
    cur_x = x_min
    going_up = True

    # Forward pass: from x_min to x_max
    while cur_x <= x_max:
        if going_up:
            waypoints.append((cur_x, y_max))
        else:
            waypoints.append((cur_x, y_min))
        cur_x += x_step
        going_up = not going_up

    # Step back to the final valid x
    cur_x -= x_step
    going_up = not going_up

    # Return pass: from x_max back to x_min
    while cur_x >= x_min:
        if going_up:
            waypoints.append((cur_x, y_max))
        else:
            waypoints.append((cur_x, y_min))
        cur_x -= x_step
        going_up = not going_up

    # Optionally end near the original bottom-left corner
    waypoints.append((x_min, y_min))

    return waypoints


###############################################################################
# ZigzagSmoothAvoidNode
#   - 1) Generates zigzag waypoints and moves the robot across them & back.
#   - 2) Smoothly slows or steers around obstacles, rather than abrupt stopping.
#   - 3) Uses HSV detection + 3D transforms for color-coded boxes with no double counting.
#   - 4) Stops at the end, prints a summary.
###############################################################################
class ZigzagSmoothAvoidNode(Node):
    def __init__(self):
        super().__init__('zigzag_smooth_avoid_node')

        ############################
        # A) GENERAL SETTINGS
        ############################
        self.real_robot = False   # True => depth is in mm, convert to meters
        self.visualise  = True    # Show OpenCV windows for debug
        self.global_frame = 'odom'     # or 'map' if using SLAM
        self.camera_frame = 'depth_link'  # typical LIMO depth link
        self.bridge = CvBridge()

        # We'll store unique objects as dicts: 
        # { 'color': 'red'|'green'|'blue', 'x':..., 'y':..., 'z':... }
        self.detected_objects = []
        # 3D anti-double-counting radius
        self.detection_threshold = 0.6

        # 2D bounding boxes from this frame (to avoid repeated detection)
        self.prev_bboxes_2d = []
        self.min_area_size = 100.0  # min contour area in pixel^2

        ############################
        # B) HSV THRESHOLDS
        ############################
        # Red is tricky: we combine two intervals for the hue wrap
        self.lower_red1 = np.array([0,   120,  70],  dtype=np.uint8)
        self.upper_red1 = np.array([10,  255, 255],  dtype=np.uint8)
        self.lower_red2 = np.array([170, 120,  70],  dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255],  dtype=np.uint8)
        
        # Example green & blue thresholds
        self.lower_green = np.array([35, 100, 100], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        self.lower_blue  = np.array([90, 100, 100], dtype=np.uint8)
        self.upper_blue  = np.array([130, 255, 255],dtype=np.uint8)

        ############################
        # C) CAMERA MODELS & TF
        ############################
        self.ccamera_model = None
        self.dcamera_model = None
        self.color2depth_aspect = None
        self.image_depth_ros = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Topics (adjust for your real or simulated robot)
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
        self.ccamera_info_sub = self.create_subscription(
            CameraInfo, ccamera_info_topic, self.ccamera_info_callback, qos.qos_profile_sensor_data)
        self.dcamera_info_sub = self.create_subscription(
            CameraInfo, dcamera_info_topic, self.dcamera_info_callback, qos.qos_profile_sensor_data)
        self.cimage_sub = self.create_subscription(
            Image, cimage_topic, self.image_color_callback, qos.qos_profile_sensor_data)
        self.dimage_sub = self.create_subscription(
            Image, dimage_topic, self.image_depth_callback, qos.qos_profile_sensor_data)

        ############################
        # D) ZIGZAG WAYPOINTS
        ############################
        self.x_min = -4.0
        self.x_max =  4.0
        self.y_min = -4.0
        self.y_max =  4.0
        self.x_step= 2.0  # spacing for each vertical pass

        self.zigzag_waypoints = create_zigzag_waypoints(
            self.x_min, self.x_max, self.y_min, self.y_max, self.x_step
        )
        self.current_waypoint_idx = 0
        self.waypoint_tolerance   = 0.3

        # Track if we are done
        self.done = False

        ############################
        # E) OBSTACLE AVOIDANCE
        ############################
        self.laser_ranges = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw= 0.0

        # If front distance < avoid_slow_threshold => slow down
        # If < avoid_stop_threshold => pivot to avoid collision
        self.avoid_slow_threshold = 0.8
        self.avoid_stop_threshold = 0.4

        self.normal_speed = 0.3
        self.slow_speed   = 0.1

        # Movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Timer: movement update ~10Hz
        self.timer_movement = self.create_timer(0.1, self.movement_loop)

        self.get_logger().info("ZigzagSmoothAvoidNode launched with slow maneuvering around obstacles.")

    #---------------------------------------------------------------------------
    # (1) Movement & Smooth Obstacle Avoidance
    #---------------------------------------------------------------------------
    def laser_callback(self, msg: LaserScan):
        self.laser_ranges = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_x = p.x
        self.current_y = p.y
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_yaw = yaw

    def movement_loop(self):
        """Main movement logic: drive to the next zigzag waypoint,
           slow or pivot if an obstacle is too close, 
           then continue. Stop & summarize when done."""
        if self.done:
            self.cmd_vel_pub.publish(Twist())
            return

        if not self.laser_ranges:
            return  # no laser data yet

        if self.current_waypoint_idx >= len(self.zigzag_waypoints):
            # All waypoints visited
            self.stop_and_summarize()
            return

        # Get the next waypoint
        (wx, wy) = self.zigzag_waypoints[self.current_waypoint_idx]
        dx = wx - self.current_x
        dy = wy - self.current_y
        dist = math.sqrt(dx*dx + dy*dy)

        # If we're close to the waypoint, move on
        if dist < self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            return

        # Build Twist
        twist = Twist()

        # Basic heading control
        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.current_yaw
        # Wrap error to [-pi, pi]
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # If obstacle is near => slow or pivot
        front_dist = self.get_front_distance()
        if front_dist < self.avoid_stop_threshold:
            # Pivot in place to avoid collision
            # Decide which side is more open => turn that way
            left_d  = self.get_left_distance()
            right_d = self.get_right_distance()
            # Turn toward the bigger side
            if right_d > left_d:
                twist.angular.z = -0.4  # turn right
            else:
                twist.angular.z =  0.4  # turn left
            twist.linear.x  =  0.0
        elif front_dist < self.avoid_slow_threshold:
            # Move slowly, steer gently around obstacle
            # We'll still attempt to move toward the waypoint but at slow speed
            twist.linear.x = self.slow_speed
            # simple P-control for yaw
            angular_speed = 1.0 * yaw_error
            angular_speed = max(-0.3, min(0.3, angular_speed))
            twist.angular.z = angular_speed
        else:
            # Normal forward
            angular_speed = 1.0 * yaw_error
            angular_speed = max(-0.4, min(0.4, angular_speed))
            twist.angular.z = angular_speed

            # Go forward if somewhat facing the waypoint
            if abs(yaw_error) < 0.5:
                twist.linear.x = self.normal_speed
            else:
                twist.linear.x = 0.0

        self.cmd_vel_pub.publish(twist)

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

    def stop_and_summarize(self):
        """Stop movement, print a summary of all detected objects, and mark done."""
        self.done = True
        self.cmd_vel_pub.publish(Twist())  # Stop

        # Summarize objects
        red_count   = sum(1 for obj in self.detected_objects if obj['color'] == 'red')
        green_count = sum(1 for obj in self.detected_objects if obj['color'] == 'green')
        blue_count  = sum(1 for obj in self.detected_objects if obj['color'] == 'blue')
        total_count = len(self.detected_objects)

        self.get_logger().info("=====================================")
        self.get_logger().info("ZIGZAG COMPLETE. FINAL OBJECT COUNTS:")
        self.get_logger().info(f"  Total: {total_count}")
        self.get_logger().info(f"   Red:   {red_count}")
        self.get_logger().info(f"   Green: {green_count}")
        self.get_logger().info(f"   Blue:  {blue_count}")
        idx = 1
        for obj in self.detected_objects:
            c = obj['color']
            x = obj['x']
            y = obj['y']
            z = obj['z']
            self.get_logger().info(f"  {idx}. {c} at ({x:.2f}, {y:.2f}, {z:.2f})")
            idx += 1
        self.get_logger().info("=====================================")

    #---------------------------------------------------------------------------
    # (2) Camera Info & Depth
    #---------------------------------------------------------------------------
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
        """Compute a ratio to approximately map color image coords -> depth image coords."""
        if self.ccamera_model and self.dcamera_model and self.color2depth_aspect is None:
            c_aspect = (math.atan2(self.ccamera_model.width, 2*self.ccamera_model.fx())
                        / self.ccamera_model.width)
            d_aspect = (math.atan2(self.dcamera_model.width, 2*self.dcamera_model.fx())
                        / self.dcamera_model.width)
            self.color2depth_aspect = c_aspect / d_aspect
            self.get_logger().info(f"color2depth_aspect = {self.color2depth_aspect:.3f}")

    def image_depth_callback(self, msg: Image):
        self.image_depth_ros = msg

    #---------------------------------------------------------------------------
    # (3) Color Detection in HSV + 3D Projection
    #---------------------------------------------------------------------------
    def image_color_callback(self, msg: Image):
        if (self.color2depth_aspect is None or
            self.image_depth_ros is None or
            not self.ccamera_model or
            not self.dcamera_model):
            return

        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")

        if self.real_robot:
            depth_img /= 1000.0  # convert mm -> m

        # Convert color to HSV
        hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # Red (two intervals)
        mask_red1 = cv2.inRange(hsv_img, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv_img, self.lower_red2, self.upper_red2)
        mask_red  = cv2.bitwise_or(mask_red1, mask_red2)

        # Green
        mask_green = cv2.inRange(hsv_img, self.lower_green, self.upper_green)
        # Blue
        mask_blue  = cv2.inRange(hsv_img, self.lower_blue, self.upper_blue)

        # Combine
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_blue)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_size:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rect2d = Rectangle2D(x, y, x+w, y+h)

            # Skip if overlaps with an existing rect in this frame
            overlap = any(rect2d.overlaps(old) for old in self.prev_bboxes_2d)
            if overlap:
                continue
            new_bboxes.append(rect2d)

            # Determine color by summing each mask in the bounding box
            roi_r = mask_red[y:y+h, x:x+w]
            roi_g = mask_green[y:y+h, x:x+w]
            roi_b = mask_blue[y:y+h, x:x+w]

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

            # Centroid
            cx = x + w/2
            cy = y + h/2

            dx, dy = self.color_to_depth_coords(cx, cy, color_img, depth_img)
            if (dx < 0 or dy < 0 or 
                dx >= depth_img.shape[1] or 
                dy >= depth_img.shape[0]):
                continue

            depth_val = depth_img[int(dy), int(dx)]
            if depth_val <= 0.0 or math.isinf(depth_val) or math.isnan(depth_val):
                continue

            # Project to 3D in camera frame
            from image_geometry import PinholeCameraModel
            ray = np.array(self.ccamera_model.projectPixelTo3dRay((cx, cy)))
            ray *= (depth_val / ray[2])

            camera_pose = Pose()
            camera_pose.position.x = ray[0]
            camera_pose.position.y = ray[1]
            camera_pose.position.z = ray[2]
            camera_pose.orientation.w = 1.0

            # TF to global frame
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

            # Anti-double-count
            if self.is_new_object(gx, gy, gz, color_label):
                self.detected_objects.append({
                    'color': color_label,
                    'x': gx, 'y': gy, 'z': gz
                })
                self.get_logger().info(f"NEW {color_label} object at ({gx:.2f}, {gy:.2f}, {gz:.2f}). "
                                       f"Total now {len(self.detected_objects)}.")
            else:
                self.get_logger().info("Object already counted. Skipping...")

            # Visualization
            if self.visualise:
                cv2.rectangle(color_img, (x,y), (x+w, y+h), (255,255,0), 2)
                cv2.putText(color_img, color_label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Update bounding boxes for this frame
        self.prev_bboxes_2d = new_bboxes

        if self.visualise:
            cv2.imshow("Color HSV detection", color_img)
            scaled_depth = depth_img / 5.0
            scaled_depth = np.clip(scaled_depth, 0.0, 1.0)
            cv2.imshow("Depth", scaled_depth)
            cv2.waitKey(1)

    def color_to_depth_coords(self, cx, cy, color_img, depth_img):
        """Approx shift from color px coords -> depth coords using color2depth_aspect."""
        color_cx = color_img.shape[1] / 2.0
        color_cy = color_img.shape[0] / 2.0
        depth_cx = depth_img.shape[1] / 2.0
        depth_cy = depth_img.shape[0] / 2.0

        shift_x = cx - color_cx
        shift_y = cy - color_cy

        dx = depth_cx + shift_x * self.color2depth_aspect
        dy = depth_cy + shift_y * self.color2depth_aspect
        return dx, dy

    def is_new_object(self, gx, gy, gz, color_label):
        """Check if there's no existing object of the same color within 
           self.detection_threshold in 3D space."""
        for obj in self.detected_objects:
            if obj['color'] == color_label:
                dx = obj['x'] - gx
                dy = obj['y'] - gy
                dz = obj['z'] - gz
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < self.detection_threshold:
                    return False
        return True


###############################################################################
# main()
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    node = ZigzagSmoothAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
