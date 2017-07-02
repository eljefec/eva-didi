import lidar as ld
import multibag as mb
import numpy as np
import numpystream
import cv2

INPUT_IMAGE='INPUT_IMAGE'
OUTPUT_ROTATION='OUTPUT_ROTATION'

def build_model():
    pass

def get_birdseye_box(lidar, obs):
    slice_config = ld.slice_config()
    slice_config.SIDE_RANGE = (-2.5, 2.5)
    slice_config.FWD_RANGE = (-2.5, 2.5)

    print (obs.position[0], obs.position[1])

    return ld.lidar_to_birdseye(lidar,
                                slice_config,
                                return_points = False,
                                center = (obs.position[0], obs.position[1]))

def generate_birdseye_boxes(multi, batch_size):
    generator = multi.generate(infinite = True)
    images = []
    rotations = []
    for numpydata in generator:
        lidar = numpydata.lidar
        obs = numpydata.obs[1]
        if lidar is not None:
            birdseye_box = get_birdseye_box(lidar, obs)

            images.append(birdseye_box)
            rotations.append(obs.yaw)

            if batch_size == len(images):

                image_batch = np.stack(images)
                rotation_batch = np.stack(rotations)

                images[:] = []
                rotations[:] = []

                yield ({INPUT_IMAGE: image_batch},
                       {OUTPUT_ROTATION: rotation_batch})

def train_rotation_detector(multi):
    generator = generate_birdseye_boxes(multi, batch_size=50)
    count = 0
    for b in generator:
        cv2.imwrite('birdseye_box_test{}.png'.format(count), b[0][INPUT_IMAGE][0])
        count += 1
        if count == 10:
            return

def detect_rotation(birdseye):
    pass

if __name__ == '__main__':
    bagdir = '/data/bags/didi-round2/release/car/training/nissan_driving_past_it'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, numpystream.generate_numpystream)
    train_rotation_detector(multi)
