import fnmatch
import os
import rosbag

def count_msgs(bag_file, topic):
    bag = rosbag.Bag(bag_file, 'r')
    count = bag.get_message_count(topic_filters=[topic])
    bag.close()

    return count

def count_image_msgs(bag_file):
    return count_msgs(bag_file, '/image_raw')

def count_velodyne_packets(bag_file):
    return count_msgs(bag_file, '/velodyne_packets')

def count_velodyne_points(bag_file):
    return count_msgs(bag_file, '/velodyne_points')

def find_files(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
    return sorted(matched_files)

def find_bags(directory):
    return find_files(directory, '*.bag')
