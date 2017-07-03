import cv2
import numpy as np
import yaml

import multibag as mb
import numpystream as ns

def read_ost_yaml():
    ost_yaml_path = '/home/eljefec/repo/didi-competition/calibration/ost.yaml'
    with open(ost_yaml_path , 'r') as f:
        return yaml.load(f)

def read_ost_array(ost, field):
    ost_cm = ost[field]
    return np.array(ost_cm['data']).reshape(ost_cm['cols'], ost_cm['rows'])

class Undistorter:
    def __init__(self):
        ost = read_ost_yaml()
        self.camera_matrix = read_ost_array(ost, 'camera_matrix')
        self.distortion_coefficients = read_ost_array(ost, 'distortion_coefficients')

    def undistort_image(self, im):
        return cv2.undistort(im, self.camera_matrix, self.distortion_coefficients)

def try_undistort(desired_count):
    undist = Undistorter()

    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    generator = multi.generate(infinite = False)
    count = 0
    output_count = 0
    for numpydata in generator:
        im = numpydata.image
        undistorted = undist.undistort_image(im)
        if count % 25 == 0:
            cv2.imwrite('/data/dev/orig_{}.png'.format(count), im)
            cv2.imwrite('/data/dev/undist_{}.png'.format(count), undistorted)
            output_count += 1
        count += 1
        if output_count == desired_count:
            return

if __name__ == '__main__':
    print(read_ost_yaml())
    try_undistort(5)
