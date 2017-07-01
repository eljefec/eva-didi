import framestream
import matplotlib.pyplot as plt
import multibag
import numpy as np
import os
import pickle

FIG_DIR = 'figs'

def make_and_save_histograms(bag, poses, dims, show = False):
    bins = 50
    poses = np.stack(poses)
    print(poses.shape)
    histograms = []

    f, axarr = plt.subplots(3, 3)
    plt.title(bag)

    for pose_dim in dims:
        pose_slice = poses[:, pose_dim]
        hist = np.histogram(pose_slice, bins=bins)
        histograms.append(hist)

        sub = axarr[pose_dim % 3][pose_dim / 3]
        sub.set_title(pose_dim)
        sub.hist(pose_slice, bins=100)

    filename = 'posehist' + bag.replace('/', '_') + '.png'
    filename = os.path.join(FIG_DIR, filename)
    f.set_size_inches(16, 10)
    f.savefig(filename, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()

    histograms = np.stack(histograms)
    return histograms

class PoseHistograms:
    def __init__(self, histograms_by_bag, histograms_all):
        self.histograms_by_bag = histograms_by_bag
        self.histograms_all = histograms_all

def make_histograms(bag_tracklets, show):
    histograms_by_bag = dict()
    all_poses = []

    dims = [i for i in range(9)]
    print('dims', dims)

    for bt in bag_tracklets:
        print(bt.bag)
        poses = []

        generator = framestream.generate_trainmsgs(bt.bag, bt.tracklet)
        for sample in generator:
            poses.append(sample.pose)

        histograms_by_bag[bt.bag] = make_and_save_histograms(bt.bag, poses, dims, show)

        all_poses += poses

        print('len(poses)', len(poses))
        print('len(all_poses)', len(all_poses))

    all_poses = np.stack(all_poses)
    print('all_poses.shape', all_poses.shape)

    histograms_all = make_and_save_histograms('all_bags', all_poses, dims, show)

    pose_histograms = PoseHistograms(histograms_by_bag, histograms_all)

    with open(os.path.join(FIG_DIR, 'posehist.p'), 'wb') as f:
        pickle.dump(pose_histograms, f)

    with open(os.path.join(FIG_DIR, 'all_poses.p'), 'wb') as f:
        pickle.dump(all_poses, f)

def print_hist():
    with open(os.path.join(FIG_DIR, 'all_poses.p'), 'rb') as f:
        all_poses = pickle.load(f)

    with open(os.path.join(FIG_DIR, 'posehist.p'), 'rb') as f:
        pose_histograms = pickle.load(f)

    dim_names = ['h', 'w', 'l', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    for i in range(len(dim_names)):
        print('Dimension: ', dim_names[i])
        print pose_histograms.histograms_all[i]

if __name__ == '__main__':
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    # bagdir = '/data/bags/'
    # bt = multibag.find_bag_tracklets(bagdir, '/data/tracklets/')
    # make_histograms(bt, show = False)

    print_hist()
