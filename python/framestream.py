import numpy as np
import rospy
import sensor
import parse_tracklet

class Pose:
    def __init__(self, size, trans, rots):
        self.h = size[0]
        self.w = size[1]
        self.l = size[2]
        self.tx = trans[0]
        self.ty = trans[1]
        self.tz = trans[2]
        self.rx = rots[0]
        self.ry = rots[1]
        self.rz = rots[2]

    def get_array(self):
        return np.array([self.h,
                         self.w,
                         self.l,
                         self.tx,
                         self.ty,
                         self.tz,
                         self.rx,
                         self.ry,
                         self.rz],
                        dtype=float)

class TrainMsg:
    def __init__(self, pose, image, lidar):
        self.pose = pose
        self.image = image
        self.lidar = lidar

        self._check_delay()

    def _check_delay(self):
        if self.lidar is not None and self.image is not None:
            diff = self.image.header.stamp - self.lidar.header.stamp
            if diff >= rospy.Duration(1):
                print('Warning: Image-lidar delay of {} secs {} nsecs. Nulling lidar'.format(diff.secs, diff.nsecs))
                self.lidar = None

def is_before(first, second):
    if first is not None and second is not None:
        return first.header.stamp <= second.header.stamp
    else:
        return True

class OrderChecker:
    def __init__(self, ordercheck):
        self.ordercheck = ordercheck
        self.prev_sample = None

    def check_sample(self, sample):
        if self.ordercheck and self.prev_sample is not None and sample is not None:
            assert(is_before(sample.lidar, sample.image))
            assert(is_before(self.prev_sample.image, sample.image))
            assert(is_before(self.prev_sample.lidar, sample.image))
            assert(is_before(self.prev_sample.lidar, sample.lidar))

        self.prev_sample = sample

# tracklet_file is allowed to be None
def generate_trainmsgs(bag_file, tracklet_file):
    prev_image = None
    prev_lidar = None
    tracklet = None
    frame = 0
    order_checker = OrderChecker(ordercheck = True)

    if tracklet_file is not None:
        tracklets = parse_tracklet.parse_xml(tracklet_file)
        assert(1 == len(tracklets))

        tracklet = tracklets[0]
        assert(0 == tracklet.first_frame)

    msg_generator = sensor.generate_sensormsgs(bag_file)

    while tracklet is None or frame < tracklet.num_frames:
        if tracklet is None:
            pose = None
        else:
            pose = Pose(tracklet.size, tracklet.trans[frame], tracklet.rots[frame])

        frame += 1

        msg = None
        while msg is None or msg.header is None or msg.header.frame_id != 'camera':
            # Allow msg_generator to raise StopIteration
            msg = next(msg_generator)

            if (msg is not None and msg.header is not None):
                if msg.header.frame_id == 'camera':
                    prev_image = msg
                elif msg.header.frame_id == 'velodyne':
                    prev_lidar = msg

        sample = TrainMsg(pose, prev_image, prev_lidar)

        order_checker.check_sample(sample)

        yield sample

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
    generator = generate_trainmsgs('/data/didi/didi-round1/Didi-Release-2/Data/1/3.bag', '/old_data/output/tracklet/1/3/tracklet_labels.xml')

    for sample in generator:
        samples.append(sample)
        print('track: {0}'.format(sample.pose))
        if sample.image is not None:
            print('image: {0}'.format(sample.image.header.stamp))
        if sample.lidar is not None:
            print('lidar: {0}'.format(sample.lidar.header.stamp))
        print('len(samples): {0}'.format(len(samples)))

    exit()

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
