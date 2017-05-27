import trainmsg

class TrainData:
    def __init__(self, image, lidar_panorama, lidar_slices):
        self.image = image
        self.lidar_panorama = lidar_panorama
        self.lidar_slices = lidar_slices

class TrainDataStream:
    def __init__(self):
        self.msgstream = trainmsg.TrainMsgStream()

    def start_read(self, bag_file, tracklet_file):
        self.msgstream.start_read(bag_file, tracklet_file)

    def empty(self):
        return self.msgstream.empty()

    # Precondition: empty() returns False
    def get_next_msg(self):
        msg = self.msgstream.get_next_msg()
        lidar_panorama = lidar.lidar_to_panorama(msg.lidar)
        lidar_slice = lidar.lidar_to_slice(msg.lidar)

        return TrainData(msg.pose, msg.image, lidar_panorama, lidar_slices)
