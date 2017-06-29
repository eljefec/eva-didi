import image as imlib
import lidar as ld
import multibag
import numpystream
import numpy as np
import os

def bbox_points(pose):
    side_length = max(pose.w, pose.l)
    half_side = side_length / 2
    return np.array([[pose.tx + half_side, pose.ty + half_side, pose.tz],
                     [pose.tx - half_side, pose.ty - half_side, pose.tz]
                    ])

def write_field(f, value):
    f.write(' ')
    f.write(value)

def write_kitti_annotation(pose, birdseye_bbox, filename):
    with open(filename, 'w') as f:
        f.write(pose.object_type)
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

def generate_kitti(bag_tracklets, imagedir, labeldir, output_bbox):
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    if not os.path.exists(labeldir):
        os.makedirs(labeldir)

    id = 0

    stream = multibag.MultiBagStream(bag_tracklets, numpystream.generate_numpystream)
    for numpydata in stream.generate(infinite = False):
        lidar = numpydata.lidar
        pose = numpydata.pose
        if lidar is not None:
            print(pose.tx, pose.ty, pose.tz)

            bbox = bbox_points(pose)

            print(bbox)

            birdseye = ld.lidar_to_birdseye(lidar)
            birdseye_bbox = ld.lidar_to_birdseye(bbox, return_points = True)

            print('birdseye_bbox.shape', birdseye_bbox.shape)
            print('birdseye_bbox', birdseye_bbox)

            if birdseye_bbox.shape[0] == 2 and birdseye_bbox.shape[1] == 2:
                if output_bbox:
                    bbox_tuple = ((birdseye_bbox[0][0], birdseye_bbox[0][1]),
                                  (birdseye_bbox[1][0], birdseye_bbox[1][1]))
                else:
                    bbox_tuple = None

                image_file = os.path.join(imagedir, '{:06d}.png'.format(id))
                imlib.save_np_image(birdseye, image_file, bbox_tuple)

                label_path = os.path.join(labeldir, '{:06d}.txt'.format(id))
                write_kitti_annotation(pose, birdseye_bbox, label_path)

                id += 1

if __name__ == '__main__':
    bagdir = '/data/bags/didi-round2/release/car/training/suburu_driving_past_it'
    bag_tracklets = multibag.find_bag_tracklets(bagdir, '/data/tracklets')
    generate_kitti(bag_tracklets,
                   '/data/KITTI/training/image',
                   '/data/KITTI/training/label',
                   output_bbox = True)
