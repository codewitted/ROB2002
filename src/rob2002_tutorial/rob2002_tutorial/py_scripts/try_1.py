import csv
import os
import rclpy
from rclpy.node import Node
from rclpy import qos
import math
import time

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray

class Counter3D(Node):
    detection_threshold = 0.2

    def __init__(self):
        super().__init__('counter_3d')
        self.detected_objects = []
        self.start_time = time.time()

        # CSV logging
        self.log_file = 'object_count_log.csv'
        self.fieldnames = ['timestamp', 'count']
        # If the file doesn't exist, create it and add headers
        if not os.path.isfile(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        # subscribe to object detector
        self.subscriber = self.create_subscription(
            PoseStamped,
            '/object_location',
            self.counter_callback,
            qos_profile=qos.qos_profile_sensor_data
        )
        
        # publish all detected objects as an array of poses
        self.publisher = self.create_publisher(
            PoseArray,
            '/object_count_array',
            qos.qos_profile_parameters
        )

    def counter_callback(self, data):
        new_object = data.pose
        object_exists = False
        for obj in self.detected_objects:
            pos_a = obj.position
            pos_b = new_object.position
            d = math.sqrt((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2 + (pos_a.z - pos_b.z) ** 2)
            if d < self.detection_threshold:
                object_exists = True
                break

        if not object_exists:
            self.detected_objects.append(new_object)

        # Publish poses
        parray = PoseArray(header=Header(frame_id=data.header.frame_id))
        parray.poses = self.detected_objects
        self.publisher.publish(parray)

        # Log the current count
        current_time = time.time() - self.start_time
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({'timestamp': current_time, 'count': len(self.detected_objects)})

        self.get_logger().info(f'Total count: {len(self.detected_objects)}')

def main(args=None):
    rclpy.init(args=args)
    counter_3d = Counter3D()
    rclpy.spin(counter_3d)
    counter_3d.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
