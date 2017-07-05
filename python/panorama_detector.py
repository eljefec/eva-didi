import cv2
import numpy as np
import os

import crop_images as ci
import generate_kitti
import image as imlib
import lidar as ld
import multibag as mb
import numpystream as ns
import util.traingen
import util.stopwatch

def generate_panoramas(numpydata_generator):
    for numpydata in numpydata_generator:
        lidar = numpydata.lidar
        obs = numpydata.obs
        if lidar is not None:
            frame_idx, obs = obs

            im = ld.lidar_to_panorama(lidar)
            bbox = ld.lidar_to_panorama(obs.get_bbox().transpose(), return_points = True)

            bbox = bbox.transpose()
            bbox = np.array([[np.min(bbox[0]), np.min(bbox[1])],
                             [np.max(bbox[0]), np.max(bbox[1])]
                            ])

            assert(im.shape == (38, 901, 3)), 'Shape was {}'.format(im.shape)

            # Resize for squeezeDet, which requires dimensions that are multiples of 16.
            im = cv2.resize(im, (912, 48))

            assert(im.shape == (48, 912, 3)), 'Shape was {}'.format(im.shape)

            yield (im, bbox, obs)

def generate_panoramas_multi(multi):
    return generate_panoramas(multi.generate(infinite = False))

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_bbox_label_dirs(dir):
    imagedir = os.path.join(dir, 'image_2')
    labeldir = os.path.join(dir, 'label_2')

    return (labeldir, imagedir)

def get_label_path(labeldir, id):
    return util.traingen.get_example_path(labeldir, id, 'txt')

def get_image_path(imagedir, id):
    return util.traingen.get_example_path(imagedir, id, 'png')

def write_train_data(multi, outdir):
    labeldir, imagedir = get_bbox_label_dirs(outdir)

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

def try_write():
    bagdir = '/data/bags/'
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    write_train_data(multi, '/home/eljefec/repo/squeezeDet/data/KITTI/panorama912x48')

def try_draw_panoramas():
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bagdir = '/data/bags/didi-round2/release/pedestrian/'
    # bag_file = '/data/bags/didi-round2/release/car/testing/ford02.bag'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)

    # numpystream = ns.generate_numpystream(bag_file, tracklet)
    generator = generate_panoramas_multi(multi)

    id = 1
    for im, bbox, obs in generator:
        print('bbox', bbox)
        print('bbox[0]', bbox[0])
        cv2.rectangle(im, tuple(bbox[0]), tuple(bbox[1]), color = (255, 0, 0))
        im = cv2.resize(im, (0,0), fx=1.0, fy=8.0)
        plt.imshow(im)
        plt.show()

if __name__ == '__main__':
    try_write()
    # try_draw_panoramas()
