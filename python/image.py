import cv2
import numpy as np
import os
from PIL import Image
import rosbag

def save_np_image(nparr, relative):
    im = Image.fromarray(nparr)
    full_path = os.path.join('/data/output/', relative)
    im.save(full_path)

def read_images(msg_count = None):
    data_dir = '/data/Didi-Release-2/Data/1'
    bag_name = '2.bag'
    bag_file = os.path.join(data_dir, bag_name)

    output_dir = '/data/output/image_raw'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bag = rosbag.Bag(bag_file, "r")
    messages = bag.read_messages(topics=["/image_raw"])
    num_images = bag.get_message_count(topic_filters=["/image_raw"])

    if msg_count is None:
        msg_count = num_images
    else:
        msg_count = min(msg_count, num_images)

    for i in range(msg_count):
        topic, msg, t  = messages.next()

        img = np.fromstring(msg.data, dtype=np.uint8)
        img = img.reshape(msg.height, msg.width)

        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
        save_np_image(img, 'image_raw/' + str(msg.header.seq) + '.png')

if __name__ == '__main__':
    read_images(5)
