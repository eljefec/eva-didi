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

            yield (im, bbox, obs)

def generate_panoramas_multi(multi):
    return generate_panoramas(multi.generate(infinite = False))

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_bbox_label_dirs(dir):
    imagedir = os.path.join(dir, 'image')
    labeldir = os.path.join(dir, 'label')

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

def try_write():
    bagdir = '/data/bags/'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    write_train_data(multi, '/home/eljefec/repo/squeezeDet/data/KITTI/panorama_train')

def try_draw_panoramas():
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    imagedir = '/data/dev/panorama_train'
    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_at_distance'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    generator = generate_panoramas(multi.generate(infinite = False))

    id = 1
    for im, bbox in generator:
        print('bbox', bbox)
        # image_file = os.path.join(imagedir, '{:06d}.png'.format(id))
        # imlib.save_np_image(im, image_file, bbox)
        cv2.rectangle(im, bbox[0], bbox[1], color = (255, 0, 0))
        plt.imshow(im)
        plt.show()

if __name__ == '__main__':
    try_write()
    # try_draw_panoramas()
