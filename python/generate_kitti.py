import image as imlib
import lidar as ld
import multibag
import numpystream
import numpy as np

def bbox_points(pose):
    side_length = max(pose.w, pose.l)
    half_side = side_length / 2
    return np.array([[pose.tx - half_side, pose.ty - half_side, pose.tz],
                     [pose.tx + half_side, pose.ty + half_side, pose.tz]])

def generate_kitti(bag_tracklets):
    stream = multibag.MultiBagStream(bag_tracklets, numpystream.generate_numpystream)
    for numpydata in stream.generate():
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
                bbox_tuple = ((birdseye_bbox[0][0], birdseye_bbox[0][1]),
                              (birdseye_bbox[1][0], birdseye_bbox[1][1]))
                imlib.save_np_image(birdseye,
                                    'birdseye_test.png',
                                    bbox_tuple)
                exit()

if __name__ == '__main__':
    bagdir = '/data/bags/didi-round2/release/car/training/suburu_driving_past_it'
    bag_tracklets = multibag.find_bag_tracklets(bagdir, '/data/tracklets')
    generate_kitti(bag_tracklets)
