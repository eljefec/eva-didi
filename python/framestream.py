import numpy as np
import rospy
import sensor
import parse_tracklet

class TrainMsg:
    def __init__(self, pose, image, lidar):
        self.pose = pose
        self.image = image
        self.lidar = lidar

def is_before(first, second):
    if first is not None and second is not None:
        return first.header.stamp <= second.header.stamp
    else:
        return True

class OrderChecker:
    def __init__(self, ordercheck, delaycheck):
        self.ordercheck = ordercheck
        self.delaycheck = delaycheck
        self.prev_sample = None

    def check_delay(self, lidar, image):
        if lidar is not None and image is not None:
            diff = image.header.stamp - lidar.header.stamp
            if diff >= rospy.Duration(1):
                print('Warning: Image-lidar delay of {} secs {} nsecs'.format(diff.secs, diff.nsecs))
            assert (diff < rospy.Duration(3)), "Image-lidar delay of {} secs {} nsecs".format(diff.secs, diff.nsecs)

    def check_sample(self, sample):
        if self.delaycheck and sample is not None:
            self.check_delay(sample.lidar, sample.image)

        if self.ordercheck and self.prev_sample is not None and sample is not None:
            assert(is_before(sample.lidar, sample.image))
            assert(is_before(self.prev_sample.image, sample.image))
            assert(is_before(self.prev_sample.lidar, sample.image))
            assert(is_before(self.prev_sample.lidar, sample.lidar))

        self.prev_sample = sample

class FrameStream:
    def __init__(self):
        self.msg_queue = sensor.SensorMsgQueue(maxsize = 10, hertz = 10)
        self.reset()

    def reset(self):
        self.prev_image = None
        self.prev_lidar = None
        self.tracklet = None
        self.frame = 0
        self.order_checker = OrderChecker(ordercheck = True, delaycheck = True)

    # tracklet_file is allowed to be None
    def start_read(self, bag_file, tracklet_file):
        if not self.empty():
            raise RuntimeError('Cannot start read because read is in progress.')

        self.reset()

        if tracklet_file is not None:
            tracklets = parse_tracklet.parse_xml(tracklet_file)
            assert(1 == len(tracklets))

            self.tracklet = tracklets[0]
            assert(0 == self.tracklet.first_frame)

        self.msg_queue.start_read(bag_file, warmup_timeout_secs = 15)

    def empty(self):
        return ((self.tracklet is not None and self.frame >= self.tracklet.num_frames)
                or self.msg_queue.empty())

    # Precondition: empty() returns False
    def next(self):
        if self.tracklet is None:
            track = None
        else:
            # track: size, trans, rots
            track = np.zeros(9, dtype=float)
            # size
            for i in range(3):
                track[i] = self.tracklet.size[i]
            # trans
            for i in range(3):
                track[3 + i] = self.tracklet.trans[self.frame][i]
            # Let rotations (rots) be zero.

        self.frame += 1

        msg = None
        while msg is None or msg.header is None or msg.header.frame_id != 'camera':
        # while msg is None or msg.header.frame_id != 'camera':
            msg = self.msg_queue.next()
            if (msg is not None
                    and msg.header is not None):
                if msg.header.frame_id == 'camera':
                    self.prev_image = msg
                elif msg.header.frame_id == 'velodyne':
                    self.prev_lidar = msg

        sample = TrainMsg(track, self.prev_image, self.prev_lidar)

        self.order_checker.check_sample(sample)

        return sample

if __name__ == '__main__':
    samples = []
    msgstream = FrameStream()
    for i in range(2):
        msgstream.start_read('/data/Didi-Release-2/Data/1/3.bag', '/data/output/tracklet/1/3/tracklet_labels.xml')
        while not msgstream.empty():
            sample = msgstream.next()
            samples.append(sample)
            print('track: {0}'.format(sample.pose))
            if sample.image is not None:
                print('image: {0}'.format(sample.image.header.stamp))
            if sample.lidar is not None:
                print('lidar: {0}'.format(sample.lidar.header.stamp))
            print('len(samples): {0}'.format(len(samples)))
