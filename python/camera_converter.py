import cv2
import numpy as np
import yaml

def read_ost_yaml():
    ost_yaml_path = '/home/eljefec/repo/didi-competition/calibration/ost.yaml'
    with open(ost_yaml_path , 'r') as f:
        return yaml.load(f)

def read_ost_array(ost, field):
    ost_cm = ost[field]
    return np.array(ost_cm['data']).reshape(ost_cm['rows'], ost_cm['cols'])

def lidar_point_to_camera_origin(point):
    # Based on https://github.com/udacity/didi-competition/blob/master/mkz-description/mkz.urdf.xacro
    return (point - [1.9304 - 1.5494, 0, 0.9398 - 1.27])

class CameraConverter:
    def __init__(self):
        ost = read_ost_yaml()
        self.camera_matrix = read_ost_array(ost, 'camera_matrix')
        self.distortion_coefficients = read_ost_array(ost, 'distortion_coefficients')
        self.projection_matrix = read_ost_array(ost, 'projection_matrix')

    def undistort_image(self, im):
        return cv2.undistort(im, self.camera_matrix, self.distortion_coefficients)

    def project_point(self, lidar_point):
        # See formulas at http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        object_point = lidar_point_to_camera_origin(lidar_point)
        z_camera = object_point[0]
        x_camera = -object_point[1]
        y_camera = -object_point[2]
        camera_coord = np.array([x_camera, y_camera, z_camera, 1.0]).transpose()
        image_point = np.dot(self.projection_matrix, camera_coord)
        image_point /= image_point[2]
        return image_point

    def project_points(self, obj_points):
        count = obj_points.shape[0]
        img_points = np.zeros((count, 3))
        for i in range(count):
            img_points[i] = self.project_point(obj_points[i])
        return img_points

    def obstacle_is_in_view(self, obs):
        img_width = 1368
        img_height = 1096

        img_points = self.project_points(obs.get_bbox().transpose())

        x_points = img_points[:, 0]
        y_points = img_points[:, 1]
        x_filt = np.logical_and((x_points >= 0), (x_points < img_width))
        y_filt = np.logical_and((y_points >= 0), (y_points < img_height))
        filter = np.logical_and(x_filt, y_filt)
        indices = np.argwhere(filter).flatten()

        print('indices', indices)

        return (len(indices) >= 1)
