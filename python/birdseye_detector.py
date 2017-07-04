import numpy as np

import squeezedet as sd
import rotation_detector as rd
import lidar as ld
import crop_images as ci

class BirdsEyeDetector:
  def __init__(self):
    self.squeezedet = sd.SqueezeDetector(demo_net = 'didi')
    self.rotation_detector = rd.get_latest_detector()

  def detect(self, lidar):
    birdseye = ld.lidar_to_birdseye(lidar, ld.slice_config())

    mc = self.squeezedet.mc
    im = ci.crop_image(birdseye,
                       (mc.IMAGE_WIDTH + 1, mc.IMAGE_HEIGHT + 1, 3),
                       (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 3))

    found_car = None
    found_ped = None
    boxes, probs, classes = self.squeezedet.detect(im)
    # Assume decreasing order of probability
    for box, prob, class_idx in zip(boxes, probs, classes):
      global_box = ld.birdseye_to_global(box, ld.slice_config())

      car_box = rd.get_birdseye_box(lidar, (global_box[0], global_box[1]))
      predicted_yaw = self.rotation_detector.detect_rotation(car_box)

      sd.correct_global(global_box, class_idx)

      pose = np.array([global_box[0], global_box[1], -0.9, predicted_yaw])

      if found_car is None and class_idx == sd.CAR_CLASS:
        found_car = pose

      if found_ped is None and class_idx == sd.PED_CLASS:
        found_ped = pose

      if found_car is not None and found_ped is not None:
        break
    return (found_car, found_ped)
