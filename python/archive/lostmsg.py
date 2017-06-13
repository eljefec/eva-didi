import lidar
import my_bag_utils as bu
import os
import rosbag
import threading
import time

class LostMessageFinder:
    def __init__(self):
        self.processor = lidar.PointCloudProcessor(hertz = 5)
        self.processor.add_subscriber(self.on_lidar_msg)

        self.points_count = 0
        self.lock = threading.Lock()
        self.message_found = dict()

    def find_original(self, input_bag):
        bag = rosbag.Bag(input_bag, "r")
        messages = bag.read_messages(topics=["/velodyne_packets"])
        msg_count = bag.get_message_count(topic_filters=["/velodyne_packets"])

        print('original', msg_count)

        for i in range(msg_count):
            topic, msg, t = messages.next()
            self.message_found[msg.header.seq] = False

        bag.close()

    def find(self, input_bag):
        self.find_original(input_bag)
        self.processor.read_bag(input_bag)

        # print(self.message_found)
        print(len(self.message_found))

        print('Missing messages:')

        missing_count = 0
        for seq, found in self.message_found.iteritems():
            if not found:
                print(seq)
                missing_count += 1

        print('missing_count', missing_count)

    def on_lidar_msg(self, msg):
        with self.lock:
            self.message_found[msg.header.seq] = True
            self.points_count += 1
        print(self.points_count)

if __name__ == '__main__':
    data_dir = '/data/Didi-Release-2/Data/1'
    bag_name = '2.bag'
    bag_file = os.path.join(data_dir, bag_name)
    writer = LostMessageFinder()
    writer.find(bag_file)
