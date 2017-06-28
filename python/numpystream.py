import framestream
import image
import numpy as np
import sensor_msgs.point_cloud2 as pc2

class NumpyData:
    def __init__(self, pose, image, lidar):
        self.pose = pose
        self.image = image
        self.lidar = lidar

    def __str__(self):
        return 'image.shape: {}, image.type: {}, lidar shape: {} type: {}'.format(
                self.image.shape,
                self.image.dtype,
                self.lidar.shape if self.lidar is not None else 'None',
                self.lidar.dtype if self.lidar is not None else 'None')

def generate_numpystream(bag_file, tracklet_file):
    data_generator = framestream.generate_trainmsgs(bag_file, tracklet_file)

    while True:
        frame = next(data_generator)

        if frame.lidar is None:
            points = None
        else:
            points = pc2.read_points(frame.lidar)
            points = np.array(list(points))

        imagemsg = image.ImageMsg(frame.image)

        yield NumpyData(frame.pose, imagemsg.bgr, points)

if __name__ == '__main__':
    generator = generate_numpystream('/data/bags/didi-round1/Didi-Release-2/Data/1/3.bag', '/data/tracklets/didi-round1_Didi-Release-2_Data_1-3/tracklet_labels.xml')

    count = 0
    for datum in generator:
        count += 1
        print(datum)
    print('count: {0}'.format(count))
