import image
import lidar
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import trainmsg

class TrainData:
    def __init__(self, pose, image, lidar_panorama, lidar_slices):
        self.pose = pose
        self.image = image
        self.lidar_panorama = lidar_panorama
        self.lidar_slices = lidar_slices

    def __str__(self):
        return 'image.shape: {}, image.type: {}, lidar_panorama shape: {} type: {}, lidar_slices shape: {} type: {}'.format(
                self.image.shape,
                self.image.dtype,
                self.lidar_panorama.shape if self.lidar_panorama is not None else 'None',
                self.lidar_panorama.dtype if self.lidar_panorama is not None else 'None',
                self.lidar_slices.shape if self.lidar_slices is not None else 'None',
                self.lidar_slices.dtype if self.lidar_slices is not None else 'None')

class TrainDataStream:
    def __init__(self):
        self.msgstream = trainmsg.TrainMsgStream()

    def start_read(self, bag_file, tracklet_file):
        self.msgstream.start_read(bag_file, tracklet_file)

    def empty(self):
        return self.msgstream.empty()

    # Precondition: empty() returns False
    def next(self):
        msg = self.msgstream.next()
        if msg.lidar is None:
            lidar_panorama = None
            lidar_slices = None
        else:
            points = pc2.read_points(msg.lidar)
            points = np.array(list(points))
            lidar_panorama = lidar.lidar_to_panorama(points)
            lidar_slices = lidar.lidar_to_slices(points)

        imagemsg = image.ImageMsg(msg.image)

        return TrainData(msg.pose, imagemsg.bgr, lidar_panorama, lidar_slices)

if __name__ == '__main__':
    count = 0
    datastream = TrainDataStream()
    for i in range(2):
        datastream.start_read('/data/Didi-Release-2/Data/1/2.bag', '/data/output/test/2/tracklet_labels.xml')
        while not datastream.empty():
            td = datastream.next()
            count += 1
            print(td)
    print('count: {0}'.format(count))
