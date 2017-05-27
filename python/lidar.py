import image as imlib
import numpy as np
import os
from PIL import Image
import Queue
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pc2
import transform_points as tp
from velodyne_msgs.msg import VelodyneScan
from sensor_msgs.msg import PointCloud2

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
    import time
    time.sleep(3)
    return accumulator.msgs

def process_pc2(msg):
    # CONVERT MESSAGE TO A NUMPY ARRAY OF POINT CLOUDS
    # creates a Nx5 array: [x, y, z, reflectance, ring]
    lidar = pc2.read_points(msg)
    lidar = np.array(list(lidar))
    print(msg.header.seq)
    birds_eye = tp.birds_eye_point_cloud(lidar,
                                         side_range=(-10, 10),
                                         fwd_range=(-10, 10),
                                         res=0.1)
    birds_eye.save('/data/output/birds_eye/' + str(msg.header.seq) + '.png')

    birds_eye2 = tp.point_cloud_2_birdseye(lidar,
                                           res=0.1,
                                           side_range=(-10, 10),
                                           fwd_range=(-10, 10),
                                           height_range=(-2, 2))

    imlib.save_np_image(birds_eye2, 'birds_eye2/' + str(msg.header.seq) + '.png')

    # These values are for Velodyne HDL-32E.
    panorama = tp.point_cloud_to_panorama(lidar,
                                            v_res = 1.33,
                                            h_res = 0.4,
                                            v_fov = (-30.67, 10.67),
                                            d_range = (0, 100),
                                            y_fudge = 3)

    imlib.save_np_image(panorama, 'panorama/' + str(msg.header.seq) + '.png')

    slices = tp.birds_eye_height_slices(lidar,
                                        n_slices=8,
                                        height_range=(-2.0, 0.27),
                                        side_range=(-10, 10),
                                        fwd_range=(0, 20),
                                        res=0.1)

    # VISUALISE THE SEPARATE LAYERS IN MATPLOTLIB
    import matplotlib.pyplot as plt
    dpi = 100       # Image resolution
    fig, axes = plt.subplots(2, 4, figsize=(600/dpi, 300/dpi), dpi=dpi)
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        ax.imshow(slices[:,:,i], cmap="gray", vmin=0, vmax=255)
        ax.set_axis_bgcolor((0, 0, 0))  # Set regions with no points to black
        ax.xaxis.set_visible(False)     # Do not draw axis tick marks
        ax.yaxis.set_visible(False)     # Do not draw axis tick marks
        ax.set_title(i, fontdict={"size": 10, "color":"#FFFFFF"})
        fig.subplots_adjust(wspace=0.20, hspace=0.20)

    fig.savefig('/data/output/slices/' + str(msg.header.seq) + '.png')
    plt.close(fig)

class PointCloudProcessor:
    # If token_queue is provided, then it limits message publishing. This will publish a message for each token in token_queue.
    def __init__(self, hertz = 10, token_queue = None):
        rospy.init_node('PointCloudProcessor', anonymous=True)
        self.rate = rospy.Rate(hertz)
        self.token_queue = token_queue

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
        pub = rospy.Publisher('/velodyne_packets', VelodyneScan, queue_size=200)

        if msg_count is None:
            msg_count = n_lidar
        else:
            msg_count = min(msg_count, n_lidar)

        print('msg_count: ', msg_count)

        published_count = 0
        for i in range(msg_count):
            if self.token_queue is not None:
                token = self.token_queue.get(block = True)
            topic, msg, t = messages.next()
            pub.publish(msg)
            published_count += 1
            if self.token_queue is None:
                self.rate.sleep()

        print('published_count: {0}'.format(published_count))

TOKEN = 1234

import threading
class MessageCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
        self.token_queue = Queue.Queue()
        for i in range(200):
            self.token_queue.put(TOKEN)

    def on_msg(self, msg):
        with self.lock:
            self.count += 1
        # print('Message Count: {0}'.format(self.count))
        print('cnt: {0} header: {1}'.format(self.count, msg.header))
        self.token_queue.put(TOKEN)

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

            panorama = tp.point_cloud_to_panorama(lidar,
                                                    v_res = 1.33,
                                                    h_res = 0.4,
                                                    v_fov = (-30.67, 10.67),
                                                    d_range = (0, 100),
                                                    y_fudge = 3)
            with open('panorama.p', 'wb') as f:
                pickle.dump(panorama, f)

            slices = tp.birds_eye_height_slices(lidar,
                                                n_slices=8,
                                                height_range=(-2.0, 0.27),
                                                side_range=(-10, 10),
                                                fwd_range=(0, 20),
                                                res=0.1)

            with open('slices.p', 'wb') as f:
                pickle.dump(slices, f)

if __name__ == '__main__':
    data_dir = '/data/Didi-Release-2/Data/1'
    bag_name = '2.bag'
    bag_file = os.path.join(data_dir, bag_name)

    # pointclouds = read_pointclouds(bag_file)
    # print('len(pointclouds): {0}'.format(len(pointclouds)))

    counter = MessageCounter()
    pickler = MessagePickler()
    processor = PointCloudProcessor(token_queue = counter.token_queue)
    # processor.add_subscriber(process_pc2)
    processor.add_subscriber(counter.on_msg)
    # processor.add_subscriber(pickler.on_msg)

    # Read rosbag.
    processor.read_bag(bag_file)

    processor.spin()
