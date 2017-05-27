import numpy as np
import sensor
import parse_tracklet

class TrainSample:
    def __init__(self, pose, image, lidar):
        self.pose = pose
        self.image = image
        self.lidar = lidar

class OrderChecker:
    def __init__(self):
        self.prev_sample = None

    def check_sample(self, sample):
        if self.prev_sample is not None:
            assert(sample.lidar.stamp <= sample.image.stamp)

            assert(prev_sample.image.stamp <= sample.image.stamp)
            assert(prev_sample.lidar.stamp <= sample.lidar.stamp)
            assert(prev_sample.image.stamp <= sample.lidar.stamp)
            assert(prev_sample.lidar.stamp <= sample.image.stamp)

            self.prev_sample = sample

class DataPrep:
    def __init__(self):
        self.msg_queue = sensor.SensorMsgQueue(maxsize = 10, hertz = 10)
        self.prev_image = None
        self.prev_lidar = None
        self.tracklet = None
        self.frame = 0
        self.order_checker = OrderChecker()

    def start_read(self, bag_file, tracklet_file):
        tracklets = parse_tracklet.parse_xml(tracklet_file)
        assert(1 == len(tracklets))

        self.tracklet = tracklets[0]
        assert(0 == self.tracklet.first_frame)
        self.msg_queue.start_read(bag_file)

    def empty(self):
        return self.frame >= self.tracklet.num_frames or self.msg_queue.empty()

    # Precondition: empty() returns False
    def get_next_train_sample(self):
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
        while msg is None or msg.header.frame_id != 'camera':
            msg = self.msg_queue.next()
            if msg.header.frame_id == 'camera':
                self.prev_image = msg
            elif msg.header.frame_id == 'velodyne':
                self.prev_lidar = msg

        sample = TrainSample(track, self.prev_image, self.prev_lidar)

        self.order_checker.check_sample(sample)

        return sample

if __name__ == '__main__':
    dataprep = DataPrep()
    dataprep.start_read('/data/Didi-Release-2/Data/1/2.bag', '/data/output/test/2/tracklet_labels.xml')
    samples = []
    while not dataprep.empty():
        sample = dataprep.get_next_train_sample()
        samples.append(sample)
        print('track: {0}'.format(sample.pose))
        if sample.image is not None:
            print('image: {0}'.format(sample.image.header.stamp))
        if sample.lidar is not None:
            print('lidar: {0}'.format(sample.lidar.header.stamp))
        print('len(samples): {0}'.format(len(samples)))
