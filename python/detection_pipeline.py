import numpy as np

import birdseye_detector as bd
import camera_detector as cd
import kalman_filter as kf
import util.stopwatch as sw

class DetectionPipeline:
    def __init__(self, enable_birdseye, enable_camera, enable_kalman):
        if enable_birdseye:
            self.birdseye_detector = bd.BirdsEyeDetector()
        else:
            self.birdseye_detector = None

        if enable_camera:
            self.camera_detector = cd.CameraDetector()
        else:
            self.camera_detector = None

        if enable_kalman:
            self.car_kf = kf.KalmanFilter()
            self.ped_kf = kf.KalmanFilter()
        else:
            self.car_kf = None
            self.ped_kf = None

        self.prev_car = np.array([0., 0., -0.9, 0.])
        self.prev_ped = np.array([0., 0., -0.9, 0.])

        self.lidar_lost_car = False
        self.lidar_lost_ped = False

    def detect_lidar(self, lidar, t):
        stopwatch = sw.Stopwatch()
        if self.birdseye_detector is not None:
            car, ped = self.birdseye_detector.detect(lidar)
            self._add_detection(car, ped, t)
            self.lidar_lost_car = (car is None)
            self.lidar_lost_ped = (ped is None)
        stopwatch.stop()
        print('Detection time (secs): {}', stopwatch.format_duration(coarse = False))

    def detect_image(self, image, t):
        if self.camera_detector is not None:
            car, ped = self.camera_detector.detect(image)
            if self.lidar_lost_car and self.lidar_lost_ped:
                self._add_detection(car, ped, t)

    def _add_detection(self, car, ped, t):
        if self.car_kf is not None:
            if car is None:
                self.car_kf.predict(t)
            else:
                self.car_kf.update(car[0:2], t)
            if ped is None:
                self.ped_kf.predict(t)
            else:
                self.ped_kf.update(ped[0:2], t)

        self.prev_car = car
        self.prev_ped = ped

    def estimate_positions(self):
        if self.car_kf is not None:
            car_pose = get_pose(self.car_kf, self.prev_car)
            ped_pose = get_pose(self.ped_kf, self.prev_ped)
            return car_pose, ped_pose
        else:
            return self.prev_car, self.prev_ped

def get_pose(kf, prev_pose):
    state = kf.get_state()
    if prev_pose is None:
        tz = -0.9
    else:
        tz = prev_pose[2]
    return np.array([state.x, state.y, tz, state.yaw])
