import cv2
import math
import numpy as np
import os
import random

import crop_images as ci
import generate_kitti
import image as imlib
import lidar as ld
import multibag as mb
import numpystream as ns
import util.traingen
import util.stopwatch

OLD_SHAPE = (38, 901, 3)
NEW_SHAPE = (48, 912, 3)

def rotate(xy, rot):
    x = xy[0]
    y = xy[1]
    xy[0] = x * math.cos(rot) - y * math.sin(rot)
    xy[1] = x * math.sin(rot) + y * math.cos(rot)

def calc_width_change(bbox_orig, bbox_rotated):
    diff_orig = bbox_orig[1] - bbox_orig[0]
    diff_rotated = bbox_rotated[1] - bbox_rotated[0]

    return diff_rotated[0] - diff_orig[0]

def rotation_is_safe(obs, rot):
    bbox_points_orig = ld.lidar_to_panorama(obs.get_bbox_noncached().transpose(), return_points = True)
    rotate(obs.position, rot)
    bbox_points_rotated = ld.lidar_to_panorama(obs.get_bbox_noncached().transpose(), return_points = True)
    rotate(obs.position, -rot)

    bbox_orig = summarize_bbox(bbox_points_orig)
    bbox_rotated = summarize_bbox(bbox_points_rotated)

    width_change = calc_width_change(bbox_orig, bbox_rotated)
    if width_change > 5:
        return False
    else:
        return True

def augment_example(lidar, obs):
    rot = random.uniform(-math.pi, math.pi)

    if not rotation_is_safe(obs, rot):
        return

    rotate(obs.position, rot)

    for i in range(lidar.shape[0]):
        rotate(lidar[i], rot)

def resize_bbox(bbox, old_shape, new_shape):
    bbox[0] = bbox[0] * new_shape[1] / old_shape[1]
    bbox[1] = bbox[1] * new_shape[0] / old_shape[0]

def resize(im, bbox):
    old_shape = OLD_SHAPE
    new_shape = NEW_SHAPE

    assert(im.shape == old_shape), 'Shape was {}'.format(im.shape)

    # Resize for squeezeDet, which requires dimensions that are multiples of 16.
    im = cv2.resize(im, (new_shape[1], new_shape[0]))

    assert(im.shape == new_shape), 'Shape was {}'.format(im.shape)

    for i in range(bbox.shape[0]):
        resize_bbox(bbox[i], old_shape, new_shape)

    return im, bbox

def summarize_bbox(bbox):
    bbox = bbox.transpose()
    bbox = np.array([[np.min(bbox[0]), np.min(bbox[1])],
                     [np.max(bbox[0]), np.max(bbox[1])]
                    ])
    return bbox

def clip_bbox(bbox):
    for point in bbox:
        if point[1] < 0:
            point[1] = 0
        elif point[1] >= NEW_SHAPE[0]:
            point[1] = NEW_SHAPE[0] - 1

def generate_panoramas(numpydata_generator):
    for numpydata in numpydata_generator:
        lidar = numpydata.lidar
        obs = numpydata.obs
        if lidar is not None:
            frame_idx, obs = obs

            augment_example(lidar, obs)

            im = ld.lidar_to_panorama(lidar)
            bbox_points = ld.lidar_to_panorama(obs.get_bbox().transpose(), return_points = True)

            bbox = summarize_bbox(bbox_points)

            # Resize for squeezeDet, which requires dimensions that are multiples of 16.
            im, bbox = resize(im, bbox)

            clip_bbox(bbox)

            yield (im, bbox, obs)

def generate_panoramas_multi(multi):
    return generate_panoramas(multi.generate(infinite = False))

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_image_label_dirs(dir):
    imagedir = os.path.join(dir, 'image_2')
    labeldir = os.path.join(dir, 'label_2')

    return (labeldir, imagedir)

def get_label_path(labeldir, id):
    return util.traingen.get_example_path(labeldir, id, 'txt')

def get_image_path(imagedir, id):
    return util.traingen.get_example_path(imagedir, id, 'png')

def write_train_data(multi, outdir):
    labeldir, imagedir = get_image_label_dirs(outdir)

    makedir(labeldir)
    makedir(imagedir)

    id = 0
    generator = generate_panoramas_multi(multi)
    for im, bbox, obs in generator:
        image_path = get_image_path(imagedir, id)
        cv2.imwrite(image_path, im)
        label_path = get_label_path(labeldir, id)
        generate_kitti.write_kitti_annotation(obs, bbox, label_path)
        id += 1
        if id % 1000 == 0:
            print('Wrote {} examples.'.format(id))

    util.traingen.write_train_val(outdir, id)

def generate_train_data(train_dir, index_file, infinite = True):
    while True:
        ids = []
        with open(os.path.join(train_dir, index_file), 'r') as f:
            for id in f:
                ids.append(int(id))

        print('len(ids)', len(ids))

        labeldir, imagedir = get_image_label_dirs(train_dir)

        for id in ids:
            image_path = get_image_path(imagedir, id)
            im = cv2.imread(image_path)

            label_path = get_label_path(labeldir, id)
            bbox = generate_kitti.read_kitti_annotation(label_path)

            yield im, bbox

        if not infinite:
            return

def explore_train_data():
    train_dir = '/home/eljefec/repo/squeezeDet/data/KITTI/training'
    generator = generate_train_data(train_dir, 'val.txt', infinite = False)
    data = None
    for im, bbox in generator:
        if data is None:
            data = np.array([bbox])
        else:
            data = np.append(data, [bbox], axis = 0)
    print('data.shape', data.shape)
    print('< zero', (data < 0).sum())

def try_write():
    bagdir = '/data/bags/'
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    write_train_data(multi, '/home/eljefec/repo/squeezeDet/data/KITTI/panorama912x48')

def make_tuple(bbox):
    return (bbox[0], bbox[1])

def try_draw_panoramas():
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    bagdir = '/data/bags/'
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    # bagdir = '/data/bags/didi-round2/release/pedestrian/'
    # bag_file = '/data/bags/didi-round2/release/car/testing/ford02.bag'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)

    # numpystream = ns.generate_numpystream(bag_file, tracklet)
    generator = generate_panoramas_multi(multi)

    id = 1
    for im, bbox, obs in generator:
        cv2.rectangle(im, tuple(bbox[0]), tuple(bbox[1]), color = (255, 0, 0))
        im = cv2.resize(im, (0,0), fx=1.0, fy=8.0)
        plt.imshow(im)
        plt.show()

if __name__ == '__main__':
    # explore_train_data()
    try_write()
    # try_draw_panoramas()
