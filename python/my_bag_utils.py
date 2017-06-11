import fnmatch
import os
import rosbag

def count_image_msgs(bag_file):
    bag = rosbag.Bag(bag_file, 'r')
    count = bag.get_message_count(topic_filters=['/image_raw'])
    bag.close()

    return count

def find_bags(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
    return sorted(matched_files)

