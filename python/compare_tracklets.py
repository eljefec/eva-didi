import numpy as np
import parse_tracklet as pt

def filter_diffs(diffs):
    diffs = diffs[diffs > -2]
    diffs = diffs[diffs < 2]
    return diffs

def stats(unfiltered_diff):
    diff = filter_diffs(unfiltered_diff)
    median = np.median(diff, axis=0)
    mean = np.mean(diff, axis=0)
    min = np.min(diff, axis=0)
    max = np.max(diff, axis=0)

    print('median', median)
    print('mean', mean)
    print('min', min)
    print('max', max)

    hist = np.histogram(diff)
    print('hist', hist)

def compare_tracklets(t1, t2):
    size_diff = t1.size - t2.size
    print('size_diff', size_diff)

    trans_diff = t1.trans - t2.trans
    rots_diff = t1.rots - t2.rots

    print('trans_diff')
    for i in range(trans_diff.shape[1]):
        print('i', i)
        stats(trans_diff[:,i])

    print('rots_diff')
    for i in range(rots_diff.shape[1]):
        print('i', i)
        stats(rots_diff[:,i])

def read_tracklet(path):
    tracklets = pt.parse_xml(path)
    assert(len(tracklets) == 1)
    return tracklets[0]

def compare_tracklet_files(path1, path2):
    print('Comparing t1 - t2')
    print('t1: {}'.format(path1))
    print('t2: {}'.format(path2))
    t1 = read_tracklet(path1)
    t2 = read_tracklet(path2)
    compare_tracklets(t1, t2)

if __name__ == '__main__':
    compare_tracklet_files('/data/tracklets/didi-round2_release_pedestrian-ped_train/tracklet_labels.xml',
                           '/data/out/err_correction/ped_train.xml')
    compare_tracklet_files('/data/tracklets/didi-round2_release_car_training_bmw_following_long-bmw02/tracklet_labels.xml',
                           '/data/out/err_correction/bmw02.xml')
    compare_tracklet_files('/data/tracklets/didi-round2_release_car_training_suburu_leading_front_left-suburu11/tracklet_labels.xml',
                           '/data/out/err_correction/suburu11.xml')
