import squeezedet as sd
import camera_detector as cd

class DetectionPipeline:
    def __init__(self, enable_birdseye, enable_camera):
        if enable_birdseye:
            self.birdseye_detector = sd.BirdsEyeDetector()
        else:
            self.birdseye_detector = None

        if enable_camera:
            self.camera_detector = cd.CameraDetector()
        else:
            self.camera_detector = None

        self.prev_car = None
        self.prev_ped = None
        self.prev_t = None

    def detect(self, lidar, t):
        if self.birdseye_detector is not None:
            car, ped = self.birdseye_detector.detect(lidar)
            add_detection(car, ped, t)

    def detect(self, image, t):
        if self.camera_detector is not None:
            car, ped = self.camera_detector.detect(image)
            add_detection(car, ped, t)

    def add_detection(car, ped, t):
        if self.prev_t is None:
            self.prev_t = t
        else:
            dt = t - self.prev_t

        self.prev_car = car
        self.prev_ped = pred

    def estimate_positions(self):
        return self.prev_car, self.prev_ped
