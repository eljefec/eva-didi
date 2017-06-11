import lidar
import my_bag_utils as bu
import os
import rosbag
import threading
import time

def get_points_filename(bag_file):
    return bag_file + '.points'

def bag_contains_points(bag):
    points_count = bu.count_velodyne_points(bag)
    packet_count = bu.count_velodyne_packets(bag)

    return points_count == packet_count and points_count > 0

def bag_contains_packets(bag):
    return bu.count_velodyne_packets(bag) > 0

def conversion_is_needed(bag):
    assert (bag_contains_packets(bag)), '{} does not contain velodyne_packets'.format(bag)

    if bag_contains_points(bag):
        print('Bag already contains velodyne_points. {}'.format(bag))
        return False

    output_file = get_points_filename(bag)
    if os.path.exists(output_file):
        points_count = bu.count_velodyne_points(output_file)
        packet_count = bu.count_velodyne_packets(bag)
        if points_count == packet_count:
            print('Separate points file already generated. {}'.format(output_file))
            return False
        else:
            print('Points file generated, but count does not match. points={}, packet={}'.format(points_count, packet_count))
            return True
    else:
        return True

class VelodynePointBagWriter:
    def __init__(self):
        self.processor = lidar.PointCloudProcessor(hertz = 12)
        self.processor.add_subscriber(self.on_lidar_msg)

        self.lock = threading.Lock()
        self.warmed_up = False

    def convert_bag(self, input_bag):
        if not conversion_is_needed(input_bag):
            return

        if not self.warmed_up:
            # Convert the first bag twice. For some reason, the first bag conversion tends to miss some messages.
            self.convert_bag_help(input_bag)
            self.warmed_up = True

        self.convert_bag_help(input_bag)

    def convert_bag_help(self, input_bag):
        output_file = get_points_filename(bag)

        self.output_bag = rosbag.Bag(output_file, 'w')
        self.processor.read_bag(input_bag)

        # Allow last few messages to flow through self.on_lidar_msg() before closing output bag.
        time.sleep(3)
        with self.lock:
            self.output_bag.close()

        packet_count = bu.count_velodyne_packets(input_bag)
        points_count = bu.count_velodyne_points(output_file)
        missing_count = packet_count - points_count

        print('Missing={}. Read={}. Wrote={}. Output={}.'.format(missing_count, packet_count, points_count, output_file))

    def on_lidar_msg(self, msg):
        with self.lock:
            self.output_bag.write('/velodyne_points', msg, msg.header.stamp)

if __name__ == '__main__':
    bags = bu.find_bags('/data/')
    writer = VelodynePointBagWriter()

    for bag in bags:
        writer.convert_bag(bag)
