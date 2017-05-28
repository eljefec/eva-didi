import cv2
import numpy as np
import os
import rosbag

class ImageMsg:
    def __init__(self, msg):
        self.header = msg.header
        self.height = msg.height
        self.width = msg.width
        img = np.fromstring(msg.data, dtype=np.uint8)
        img = img.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
        self.bgr = img

def save_np_image(nparr, fullpath):
    from PIL import Image
    im = Image.fromarray(nparr)
    im.save(fullpath)

def read_images(bag_file):
    image_msgs = []

    bag = rosbag.Bag(bag_file, "r")
    messages = bag.read_messages(topics=["/image_raw"])
    num_images = bag.get_message_count(topic_filters=["/image_raw"])

    for i in range(num_images):
        topic, msg, t  = messages.next()

        image_msgs.append(ImageMsg(msg))

    return image_msgs

def save_images(bag_file, msg_count = None):
    image_msgs = read_images(bag_file)

    output_dir = '/data/output/image_raw'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for msg in image_msgs:
        save_np_image(msg.bgr, os.path.join(output_dir, 'image_raw/' + str(msg.header.seq) + '.png'))
        count += 1
        if count == msg_count:
            return

if __name__ == '__main__':
    data_dir = '/data/Didi-Release-2/Data/1'
    bag_name = '2.bag'
    bag_file = os.path.join(data_dir, bag_name)
    save_images(bag_file, 5)
