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
    return np.array(ost_cm['data']).reshape(ost_cm['rows'], ost_cm['cols'])

class Undistorter:
    def __init__(self):
        ost = read_ost_yaml()
        self.camera_matrix = read_ost_array(ost, 'camera_matrix')
        self.distortion_coefficients = read_ost_array(ost, 'distortion_coefficients')
        self.projection_matrix = read_ost_array(ost, 'projection_matrix')

    def undistort_image(self, im):
        return cv2.undistort(im, self.camera_matrix, self.distortion_coefficients)

    def project_point(self, object_point):
        # See formulas at http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        z_camera = object_point[0]
        x_camera = -object_point[1]
        y_camera = -object_point[2]
        camera_coord = np.array([x_camera, y_camera, z_camera, 1.0]).transpose()
        image_point = np.dot(self.projection_matrix, camera_coord)
        image_point /= image_point[2]
        return image_point

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
        frame_idx, obs = numpydata.obs
        undistorted = undist.undistort_image(im)
        if count % 25 == 0:
            cv2.imwrite('/data/dev/orig_{}.png'.format(count), im)

            img_point = undist.project_point(obs.position)
            print('img_point', img_point)
            cv2.circle(undistorted, (int(img_point[0]), int(img_point[1])), radius = 5, color = (255, 0, 0), thickness=2)
            cv2.imwrite('/data/dev/undist_{}.png'.format(count), undistorted)
            output_count += 1
        count += 1
        if output_count == desired_count:
            return

if __name__ == '__main__':
    print(read_ost_yaml())
    try_undistort(5)
