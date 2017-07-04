import numpy as np

import birdseye_detector as bd
import camera_detector as cd

class DetectionPipeline:
    def __init__(self, enable_birdseye, enable_camera):
        if enable_birdseye:
            self.birdseye_detector = bd.BirdsEyeDetector()
        else:
            self.birdseye_detector = None

        if enable_camera:
            self.camera_detector = cd.CameraDetector()
        else:
            self.camera_detector = None

        self.prev_car = np.array([0., 0., -0.9, 0.])
        self.prev_ped = np.array([0., 0., -0.9, 0.])
        self.prev_t = None

    def detect_lidar(self, lidar, t):
        if self.birdseye_detector is not None:
            car, ped = self.birdseye_detector.detect(lidar)
            self._add_detection(car, ped, t)

    def detect_image(self, image, t):
        if self.camera_detector is not None:
            car, ped = self.camera_detector.detect(image)
            self._add_detection(car, ped, t)

    def _add_detection(self, car, ped, t):
        if self.prev_t is None:
            self.prev_t = t
        else:
            dt = t - self.prev_t

        self.prev_car = car
        self.prev_ped = ped

    def estimate_positions(self):
        return self.prev_car, self.prev_ped
