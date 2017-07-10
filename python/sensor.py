from lidar import PointCloudProcessor
import lidarbag as lb
import my_bag_utils as bu
import numpy
import Queue
import rosbag
from threading import Thread
import time

# Precondition: roscore and velodyne node are running.
class SensorMsgQueue:
    def __init__(self, maxsize, hertz):
        self.maxsize = maxsize
        self.sleep_secs = 1 / hertz

        self.reset(check = False)

        self.lidar_processor = PointCloudProcessor(hertz)
        self.lidar_processor.add_subscriber(self.on_lidar_msg)

    def empty(self):
        return ((self.image_thread is None
                 and self.lidar_thread is None)
             or (not self.image_thread.is_alive()
                 and not self.lidar_thread.is_alive()
                 and self.image_queue.empty()
                 and self.lidar_queue.empty()))

    def can_reset(self):
        return ((self.image_thread is None
                 and self.lidar_thread is None)
             or (not self.image_thread.is_alive()
                 and not self.lidar_thread.is_alive()))

    def reset(self, check = True):
        if check and not self.can_reset():
            raise RuntimeError('Cannot reset because threads are still alive.')

        self.image_queue = Queue.Queue()
        self.lidar_queue = Queue.Queue()

        self.next_image = None
        self.next_lidar = None

        self.image_thread = None
        self.lidar_thread = None

    # Returns messages in sequence.
    # None does not mean queue is empty. Call empty().
    def next(self):
        time.sleep(0.03)

        if self.next_image is None:
            try:
                self.next_image = self.image_queue.get_nowait()
            except Queue.Empty:
                pass

        if self.next_lidar is None:
            try:
                self.next_lidar = self.lidar_queue.get_nowait()
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
    def start_read(self, bag_file, warmup_timeout_secs = None):
        self.reset()

        self.image_thread = Thread(target=self.read_images, args=(bag_file,))
        self.image_thread.start()

        self.lidar_thread = Thread(target=self.read_lidar, args=(bag_file,))
        self.lidar_thread.start()

        sleep_secs = 3
        round_count = 0
        if warmup_timeout_secs is not None:
            round_max = int(warmup_timeout_secs / sleep_secs)
        while True:
            time.sleep(sleep_secs)
            if not self.image_queue.empty() and not self.lidar_queue.empty():
                return
            if warmup_timeout_secs is not None:
                round_count += 1
                if round_count >= round_max:
                    return

    def read_images(self, bag_file):
        bag = rosbag.Bag(bag_file, 'r')
        messages = bag.read_messages(topics=["/image_raw"])
        num_images = bag.get_message_count(topic_filters=["/image_raw"])

        for i in range(num_images):
            topic, msg, t  = messages.next()
            self.image_queue.put(msg)
            time.sleep(self.sleep_secs)

        bag.close()

    def lidar_queue_is_full(self):
        return self.lidar_queue.full()

    def on_lidar_msg(self, msg):
        self.lidar_queue.put(msg)

    def read_lidar(self, bag_file):
        self.lidar_processor.read_bag(bag_file)
        # self.lidar_processor.spin()

def generate_msgs_bag(bag_file, topics):
    with rosbag.Bag(bag_file, 'r') as bag:
        messages = bag.read_messages(topics=topics)
        msg_count = bag.get_message_count(topic_filters=topics)

        for i in range(msg_count):
            topic, msg, t  = messages.next()
            yield msg

class BagMsgQueue:
    def __init__(self, bag_file, topics):
        self.generator = generate_msgs_bag(bag_file, topics)
        self.generate_next()

    def generate_next(self):
        try:
            self.nextmsg = next(self.generator)
        except StopIteration:
            self.nextmsg = None

    def peek(self):
        return self.nextmsg

    def pop(self):
        nextmsg = self.nextmsg
        self.generate_next()
        return nextmsg

def pop_next_msg(bag_msg_queues):
    smallest_i = 0
    smallest_stamp = bag_msg_queues[smallest_i].peek().header.stamp
    for i in range(1, len(bag_msg_queues)):
        current_stamp = bag_msg_queues[i].peek().header.stamp
        if current_stamp < smallest_stamp:
            smallest_i = i
            smallest_stamp = current_stamp

    nextmsg = bag_msg_queues[smallest_i].pop()

    if bag_msg_queues[smallest_i].peek() is None:
        del bag_msg_queues[smallest_i]

    return nextmsg

def generate_msgs_multibag(bag_msg_queues):
    while bag_msg_queues:
        yield pop_next_msg(bag_msg_queues)

def generate_sensormsgs(bag_file):
    assert (not lb.conversion_is_needed(bag_file)), 'Conversion is needed for {}'.format(bag_file)

    if lb.bag_contains_points(bag_file):
        topics = ['/image_raw', '/velodyne_points']
        return generate_msgs_bag(bag_file, topics)
    else:
        points_bag = lb.get_points_filename(bag_file)

        bag_msg_queues = []
        bag_msg_queues.append(BagMsgQueue(bag_file, ['/image_raw']))
        bag_msg_queues.append(BagMsgQueue(points_bag, ['/velodyne_points']))

        return generate_msgs_multibag(bag_msg_queues)

if __name__ == '__main__':
    msg_counts = dict()
    generator = generate_sensormsgs('/data/didi/didi-round1/Didi-Release-2/Data/1/10.bag')
    for msg in generator:
        # print('DEBUG: Got msg. {0} {1} {2}'.format(msg.header.stamp, msg.header.seq, msg.header.frame_id))
        key = msg.header.frame_id
        if key not in msg_counts:
            msg_counts[key] = 1
        else:
            msg_counts[key] += 1

    print(msg_counts)

### SensorMsgQueue

    msg_queue = SensorMsgQueue(maxsize = 10, hertz = 10)

    for i in range(2):
        msg_counts = dict()
        msg_queue.start_read('/data/didi/didi-round1/Didi-Release-2/Data/1/10.bag', 5)
        while not msg_queue.empty():
            msg = msg_queue.next()
            if msg is not None:
                # print('DEBUG: Got msg. {0} {1} {2}'.format(msg.header.stamp, msg.header.seq, msg.header.frame_id))
                key = msg.header.frame_id
                if key not in msg_counts:
                    msg_counts[key] = 1
                else:
                    msg_counts[key] += 1
            # if interval is not None:
            #    print('est: {0:.2f}'.format(interval))

        print(msg_counts)
