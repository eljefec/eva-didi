import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import tensorflow as tf

import detection_pipeline as dp

class RosNode:
    def __init__(self):
        rospy.init_node('EvaDetectionPipeline', anonymous=True)
        rospy.Subscriber('/velodyne_points', PointCloud2, self.pointcloud_received)
        self.pipeline = dp.DetectionPipeline(enable_birdseye = True,
                                             enable_camera = False,
                                             enable_kalman = False)

    def spin(self):
        rospy.spin()

    def pointcloud_received(self, msg):
        points = pc2.read_points(msg)
        points = np.array(list(points))
        self.pipeline.detect_lidar(points, msg.header.stamp)
        car, ped = self.pipeline.estimate_positions()
        # print('result', result)
        print('car', car)
        print('ped', ped)

def main(argv=None):
    node = RosNode()
    print('Beginning spinning...')
    node.spin()

if __name__ == '__main__':
    tf.app.run()
