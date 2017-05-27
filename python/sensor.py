import interval
from lidar import PointCloudProcessor
import numpy
import Queue
import rosbag
from threading import Thread
import time

# Precondition: roscore and velodyne node are running.
class SensorMsgQueue:
    def __init__(self, maxsize, hertz):
        self.image_queue = Queue.Queue(maxsize)
        self.lidar_queue = Queue.Queue()

        self.next_image = None
        self.next_lidar = None

        self.image_thread = None
        self.lidar_thread = None

        self.lidar_interval = interval.IntervalTracker(5)
        self.lidar_processor = PointCloudProcessor(hertz)
        self.lidar_processor.add_subscriber(self.on_lidar_msg)

    def empty(self):
        return (not self.image_thread.is_alive()
                and not self.lidar_thread.is_alive()
                and self.image_queue.empty()
                and self.lidar_queue.empty())

    # Returns messages in sequence.
    # None does not mean queue is empty. Call empty().
    def next(self):
        if self.next_image is None:
            try:
                self.next_image = self.image_queue.get_nowait()
            except Queue.Empty:
                pass

        if self.next_lidar is None:
            try:
                self.next_lidar = self.lidar_queue.get_nowait()
                self.lidar_interval.report_event()
            except Queue.Empty:
                pass

        if self.next_image is None:
            msg = self.next_lidar
            self.next_lidar = None
            return msg
        elif self.next_lidar is None:
            msg = self.next_image
            self.next_image = None
            return msg

        if self.next_image.header.stamp < self.next_lidar.header.stamp:
            msg = self.next_image
            self.next_image = None
            return msg
        else:
            msg = self.next_lidar
            self.next_lidar = None
            return msg

        return None

    # Fork threads to read from bag and fill queues.
    # Returns error if bag read is in progress.
    def start_read_bag(self, bag_file, warmup_secs):
        self.image_thread = Thread(target=self.read_images, args=(bag_file,))
        self.image_thread.start()

        self.lidar_thread = Thread(target=self.read_lidar, args=(bag_file,))
        self.lidar_thread.start()

        sleep_secs = 0.1
        for i in range(int(warmup_secs / sleep_secs)):
            time.sleep(sleep_secs)
            if not self.image_queue.empty() and not self.lidar_queue.empty():
                return

    def read_images(self, bag_file):
        bag = rosbag.Bag(bag_file, 'r')
        messages = bag.read_messages(topics=["/image_raw"])
        num_images = bag.get_message_count(topic_filters=["/image_raw"])

        print('Reading images...')

        for i in range(num_images):
            topic, msg, t  = messages.next()
            self.image_queue.put(msg)

    def lidar_queue_is_full(self):
        return self.lidar_queue.full()

    def on_lidar_msg(self, msg):
        self.lidar_queue.put(msg)

    def read_lidar(self, bag_file):
        print('Reading lidar...')

        self.lidar_processor.read_bag(bag_file)
        # self.lidar_processor.spin()

if __name__ == '__main__':
    msg_queue = SensorMsgQueue(maxsize = 10, hertz = 20)
    msg_queue.start_read_bag('/data/Didi-Release-2/Data/1/2.bag', 5)

    msg_counts = dict()

    while not msg_queue.empty():
        msg = msg_queue.next()
        if msg is not None:
            # print('DEBUG: Got msg. {0}'.format(msg.header))
            key = msg.header.frame_id
            if key not in msg_counts:
                msg_counts[key] = 1
            else:
                msg_counts[key] += 1
        # interval = msg_queue.lidar_interval.estimate_interval_secs()
        # if interval is not None:
        #    print('est: {0:.2f}'.format(interval))

    print(msg_counts)
