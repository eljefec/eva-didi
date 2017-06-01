import framestream
import matplotlib.pyplot as plt
import multibag
import numpy as np
import os

FIG_DIR = 'figs'
if __name__ == '__main__':
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    poses = []

    bag_tracklets = multibag.find_bag_tracklets('/data/Didi-Release-2/Data/', '/data/output/tracklet')
    # bag_tracklets = [multibag.BagTracklet('/data/Didi-Release-2/Data/1/3.bag', '/data/output/tracklet/1/3/tracklet_labels.xml')]

    for bt in bag_tracklets:
        print(bt)

        msgstream = framestream.FrameStream()
        msgstream.start_read(bt.bag, bt.tracklet)
        while not msgstream.empty():
            sample = msgstream.next()
            poses.append(sample.pose)
            # print('track: {0}'.format(sample.pose))
            # if sample.image is not None:
                # print('image: {0}'.format(sample.image.header.stamp))
            # if sample.lidar is not None:
                # print('lidar: {0}'.format(sample.lidar.header.stamp))

        print('len(poses): {0}'.format(len(poses)))

    poses = np.stack(poses)
    print(poses.shape)

    dims = [3, 4, 5]

    f, axarr = plt.subplots(len(dims))

    for pose_dim in dims:
        pose_slice = poses[:, pose_dim]
        print('pose_slice.shape', pose_slice.shape)
        # hist = np.histogram(pose_slice, bins=50)

        sub = axarr[pose_dim]
        sub.set_title(pose_dim)
        sub.hist(pose_slice, bins=100)

    f.savefig(FIG_DIR + '/txtytz_histogram.png', bbox_inches='tight')

    plt.show()
