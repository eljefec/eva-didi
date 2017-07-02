import crop_images as ci
import image as imlib
import lidar as ld
import multibag
import numpystream
import numpy as np
import os

def bbox_points_old(pose):
    side_length = max(pose.w, pose.l)
    half_side = side_length / 2
    return np.array([[pose.tx + half_side, pose.ty + half_side, pose.tz],
                     [pose.tx - half_side, pose.ty - half_side, pose.tz]
                    ])

def bbox_points(obs):
    bbox_3d = obs.get_bbox()
    xpts = bbox_3d[0, 0:4]
    ypts = bbox_3d[1, 0:4]
    tz = obs.position[2]
    return np.array([[np.max(xpts), np.max(ypts), tz],
                     [np.min(xpts), np.min(ypts), tz]
                    ])

def write_field(f, value):
    f.write(' ')
    f.write(value)

def write_kitti_annotation(obs, birdseye_bbox, filename):
    with open(filename, 'w') as f:
        f.write(obs.object_type)
        # Truncation
        write_field(f, '0.0')
        # Occlusion
        write_field(f, '0.0')
        # Unknown
        write_field(f, '0.0')

        flattened = birdseye_bbox.flatten()
        for value in flattened:
            write_field(f, str(value))

        # Unknown values
        for i in range(6):
            write_field(f, '0.0')

# sc is "slice_config".
def get_expected_shape(sc):
    return (1 + int((sc.SIDE_RANGE[1] - sc.SIDE_RANGE[0]) / sc.RESOLUTION),
            1 + int((sc.FWD_RANGE[1] - sc.FWD_RANGE[0]) / sc.RESOLUTION),
            3)

def assert_shape(new_shape):
    assert (new_shape[0] % 16 == 0 and new_shape[1] % 16 == 0), 'squeezeDet requires input images with dimensions divisible by 16.'

def generate_kitti(bag_tracklets, imagedir, labeldir, output_bbox, slice_config):
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    if not os.path.exists(labeldir):
        os.makedirs(labeldir)

    print('bag_tracklets', bag_tracklets)
    print('imagedir', imagedir)
    print('labeldir', labeldir)
    print('output_bbox', output_bbox)
    print('slice_config', slice_config)

    id = 0
    expected_shape = get_expected_shape(slice_config)
    new_shape = (expected_shape[0] - 1, expected_shape[1] - 1, expected_shape[2])

    assert_shape(new_shape)

    print('expected_shape', expected_shape)
    print('new_shape', new_shape)

    stream = multibag.MultiBagStream(bag_tracklets, numpystream.generate_numpystream)
    for numpydata in stream.generate(infinite = False):
        lidar = numpydata.lidar
        obs = numpydata.obs
        if lidar is not None:
            frame_idx, obs = obs
            bbox = bbox_points(obs)

            birdseye = ld.lidar_to_birdseye(lidar, slice_config)
            birdseye_bbox = ld.lidar_to_birdseye(bbox, slice_config, return_points = True)

            if birdseye_bbox.shape[0] == 2 and birdseye_bbox.shape[1] == 2:
                if output_bbox:
                    bbox_tuple = ((birdseye_bbox[0][0], birdseye_bbox[0][1]),
                                  (birdseye_bbox[1][0], birdseye_bbox[1][1]))
                else:
                    bbox_tuple = None

                crop = ci.crop_image(birdseye, expected_shape, new_shape)
                image_file = os.path.join(imagedir, '{:06d}.png'.format(id))
                imlib.save_np_image(crop, image_file, bbox_tuple)

                label_path = os.path.join(labeldir, '{:06d}.txt'.format(id))
                write_kitti_annotation(obs, birdseye_bbox, label_path)

                id += 1

if __name__ == '__main__':
    # bagdir = '/data/bags/'
    bagdir = '/data/bags/didi-round1/Didi-Release-2/Data/1/'
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_at_distance'
    bag_tracklets = multibag.find_bag_tracklets(bagdir, '/data/tracklets')
    slice_config = ld.slice_config()
    slice_config.HEIGHT_RANGE=(-1.50, 0.25)
    slice_config.SIDE_RANGE=(-40, 40)
    slice_config.FWD_RANGE=(-40, 40)
    generate_kitti(bag_tracklets,
                   '/data/KITTI_dev/rot_round1_release2/image',
                   '/data/KITTI_dev/rot_round1_release2/label',
                   output_bbox = True,
                   slice_config = slice_config)
