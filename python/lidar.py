import image as imlib
import numpy as np
import os
from PIL import Image
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pc2
import time
import transform_points as tp
from velodyne_msgs.msg import VelodyneScan
from sensor_msgs.msg import PointCloud2
from easydict import EasyDict as edict

import matplotlib
# Work around error when using matplotlib through ssh: 'Invalid DISPLAY variable'.
# matplotlib.use('agg')
import matplotlib.pyplot as plt

def lidar_to_panorama(lidar, return_points = False):
    # These values are for Velodyne HDL-32E. v_res spec is 1.33.
    return tp.point_cloud_to_panorama(lidar,
                                        v_res = 1.2,
                                        h_res = 0.4,
                                        v_fov = (-30.67, 10.67),
                                        d_range = (0, 100),
                                        y_fudge = 3,
                                        return_points = return_points)

def slice_config():
    cfg = edict()

    cfg.RESOLUTION = 0.1
    cfg.HEIGHT_RANGE=(-1.25, 0.25)
    cfg.SIDE_RANGE=(-32, 32)
    cfg.FWD_RANGE=(-32, 32)

    return cfg

def lidar_to_slices(lidar, sc):
    return tp.birds_eye_height_slices(lidar,
                                        n_slices=4,
                                        height_range = sc.HEIGHT_RANGE,
                                        side_range = sc.SIDE_RANGE,
                                        fwd_range = sc.FWD_RANGE,
                                        res = sc.RESOLUTION)

def lidar_to_birdseye(lidar, sc, return_points = False, center = (0,0)):
    return tp.point_cloud_2_birdseye(lidar,
                                     res = sc.RESOLUTION,
                                     side_range = sc.SIDE_RANGE,
                                     fwd_range = sc.FWD_RANGE,
                                     height_range = sc.HEIGHT_RANGE,
                                     return_points = return_points,
                                     center = center)

def birdseye_to_global(box, sc):
    return tp.birdseye_to_global(box[0],
                                 box[1],
                                 res = sc.RESOLUTION,
                                 side_range = sc.SIDE_RANGE,
                                 fwd_range = sc.FWD_RANGE)

class PointCloudMsg:
    def __init__(self, msg):
        lidar = pc2.read_points(msg)
        lidar = np.array(list(lidar))
        self.pointcloud = lidar
        self.header = msg.header

class PointCloudMsgAccumulator:
    def __init__(self):
        self.msgs = []

    def on_msg(self, msg):
        self.msgs.append(PointCloudMsg(msg))

def read_pointclouds(bag_file):
    accumulator = PointCloudMsgAccumulator()
    counter = MessageCounter()
    processor = PointCloudProcessor(20)
    processor.add_subscriber(accumulator.on_msg)
    processor.add_subscriber(counter.on_msg)
    processor.read_bag(bag_file)
    time.sleep(3)
    return accumulator.msgs

class PointCloudConverter:
    def __init__(self, savepath, maxcount):
        if (savepath):
            folders = ['birds_eye', 'birds_eye2', 'panorama', 'slices']
            for folder in folders:
                fullpath = os.path.join(savepath, folder)
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)
        self.savepath = savepath
        self.maxcount = maxcount
        self.count = 0
        self.lock = threading.Lock()

    def on_msg(self, msg):
        with self.lock:
            if self.maxcount is None or self.count < self.maxcount:
                convert = True
            else:
                convert = False
            self.count += 1
        if convert:
            # CONVERT MESSAGE TO A NUMPY ARRAY OF POINT CLOUDS
            # creates a Nx5 array: [x, y, z, reflectance, ring]
            lidar = pc2.read_points(msg)
            lidar = np.array(list(lidar))
            print(msg.header.seq)
            birds_eye = tp.birds_eye_point_cloud(lidar,
                                                 side_range=(-10, 10),
                                                 fwd_range=(-10, 10),
                                                 res=0.1)
            birds_eye.save(os.path.join(self.savepath, 'birds_eye/' + str(msg.header.seq) + '.png'))

            birds_eye2 = tp.point_cloud_2_birdseye(lidar,
                                                   res=0.1,
                                                   side_range=(-10, 10),
                                                   fwd_range=(-10, 10),
                                                   height_range=(-2, 2))

            imlib.save_np_image(birds_eye2, os.path.join(self.savepath, 'birds_eye2/' + str(msg.header.seq) + '.png'))

            # These values are for Velodyne HDL-32E.
            panorama = lidar_to_panorama(lidar)

            imlib.save_np_image(panorama, os.path.join(self.savepath, 'panorama/' + str(msg.header.seq) + '.png'))

            slices = lidar_to_slices(lidar, slice_config())

            # VISUALISE THE SEPARATE LAYERS IN MATPLOTLIB
            dpi = 200       # Image resolution
            fig, axes = plt.subplots(2, 2, figsize=(800/dpi, 800/dpi), dpi=dpi)
            axes = axes.flatten()
            for i,ax in enumerate(axes):
                ax.imshow(slices[:,:,i], cmap="gray", vmin=0, vmax=255)
                ax.set_facecolor('black')  # Set regions with no points to black
                ax.xaxis.set_visible(False)     # Do not draw axis tick marks
                ax.yaxis.set_visible(False)     # Do not draw axis tick marks
                ax.set_title(i, fontdict={"size": 10, "color":"#FFFFFF"})

            fig.savefig(os.path.join(self.savepath, 'slices/' + str(msg.header.seq) + '.png'), bbox_inches='tight', dpi=200)
            plt.close(fig)

class PointCloudProcessor:
    def __init__(self, hertz = 10):
        rospy.init_node('PointCloudProcessor', anonymous=True)
        self.rate = rospy.Rate(hertz)

    def add_subscriber(self, pc2_callback):
        rospy.Subscriber('/velodyne_points', PointCloud2, pc2_callback)

    def spin(self):
        rospy.spin()

    # Precondition: velodyne_pointcloud cloud_node is running.
    def read_bag(self, bag_file, msg_count = None):
        # Open rosbag.
        bag = rosbag.Bag(bag_file, "r")
        messages = bag.read_messages(topics=["/velodyne_packets"])
        n_lidar = bag.get_message_count(topic_filters=["/velodyne_packets"])

        # Publish velodyne packets from bag to topic.
        pub = rospy.Publisher('/velodyne_packets', VelodyneScan, queue_size=None)

        if msg_count is None:
            msg_count = n_lidar
        else:
            msg_count = min(msg_count, n_lidar)

        # print('msg_count: ', msg_count)

        published_count = 0
        for i in range(msg_count):
            topic, msg, t = messages.next()
            pub.publish(msg)
            published_count += 1
            self.rate.sleep()

        # print('published_count: {0}'.format(published_count))

        bag.close()

import threading
class MessageCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def on_msg(self, msg):
        with self.lock:
            self.count += 1
        # print('Message Count: {0}'.format(self.count))
        print('cnt: {0} header: {1}'.format(self.count, msg.header))

import pickle
class MessagePickler:
    def __init__(self):
        self.pickled = False
        self.lock = threading.Lock()

    def on_msg(self, msg):
        please_pickle = False
        with self.lock:
            if not self.pickled:
                please_pickle = True
                self.pickled = True
        if (please_pickle):
            with open('header.p', 'wb') as f:
                pickle.dump(msg.header, f)

            lidar = pc2.read_points(msg)
            lidar = np.array(list(lidar))
            with open('pointcloud.p', 'wb') as f:
                pickle.dump(lidar, f)

            print(lidar.shape)
            print(lidar[1:2])

            birds_eye = tp.point_cloud_2_birdseye(lidar,
                                                   res=0.1,
                                                   side_range=(-10, 10),
                                                   fwd_range=(-10, 10),
                                                   height_range=(-2, 2))
            with open('birdseye.p', 'wb') as f:
                pickle.dump(birds_eye, f)

            panorama = lidar_to_panorama(lidar)

            with open('panorama.p', 'wb') as f:
                pickle.dump(panorama, f)

            slices = lidar_to_slices(lidar, slice_config())

            with open('slices.p', 'wb') as f:
                pickle.dump(slices, f)

if __name__ == '__main__':
    data_dir = '/data/Didi-Release-2/Data/1'
    bag_name = '2.bag'
    bag_file = os.path.join(data_dir, bag_name)

    # counter = MessageCounter()
    # pickler = MessagePickler()
    converter = PointCloudConverter('/data/output', 1)
    processor = PointCloudProcessor(hertz = 40)
    # processor.add_subscriber(pickler.on_msg)
    processor.add_subscriber(converter.on_msg)
    # processor.add_subscriber(pickler.on_msg)

    # Read rosbag.
    processor.read_bag(bag_file)

    processor.spin()
