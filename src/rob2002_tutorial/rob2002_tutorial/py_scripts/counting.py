import rclpy
from rclpy.node import Node
from rclpy import qos
import math

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray

class Counter3D(Node):
    detection_threshold = 0.2 # in meters    

    def __init__(self):      
        super().__init__('counter_3d')
        
        self.detected_objects = [] # list of all detected objects

        # subscribe to object detector
        self.subscriber = self.create_subscription(PoseStamped, '/object_location', 
                                                   self.counter_callback,
                                                   qos_profile=qos.qos_profile_sensor_data)
        
        # publish all detected objects as an array of poses
        self.publisher = self.create_publisher(PoseArray, '/object_count_array',
                                               qos.qos_profile_parameters)

    def counter_callback(self, data):
        new_object = data.pose
        object_exists = False

        # check if the new object is near any previously detected object
        for existing_object in self.detected_objects:
            pos_a = existing_object.position
            pos_b = new_object.position
            d = math.sqrt((pos_a.x - pos_b.x)**2 + (pos_a.y - pos_b.y)**2 + (pos_a.z - pos_b.z)**2)
            if d < self.detection_threshold: 
                # found a close neighbour => do NOT count as a new object
                object_exists = True
                break

        # if it doesn't exist, add it
        if not object_exists:
            self.detected_objects.append(new_object)

        # publish PoseArray of all detected objects
        parray = PoseArray(header=Header(frame_id=data.header.frame_id))
        parray.poses.extend(self.detected_objects)
        self.publisher.publish(parray)

        # debug output
        print(f'total count {len(self.detected_objects)}')
        for obj in self.detected_objects:
            print(obj.position)

def main(args=None):
    rclpy.init(args=args)
    counter_3d = Counter3D()
    rclpy.spin(counter_3d)
    counter_3d.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
