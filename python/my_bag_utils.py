import rosbag

def count_image_msgs(bag_file):
    bag = rosbag.Bag(bag_file, 'r')
    count = bag.get_message_count(topic_filters=['/image_raw'])
    bag.close()

    return count
