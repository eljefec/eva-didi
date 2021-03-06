# Based loosely on https://github.com/BichenWuUCB/squeezeDet/blob/master/src/demo.py

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import moviepy.editor as mpy
import numpy as np
import tensorflow as tf

import camera_converter as cc
import crop_images as ci
import generate_tracklet
import lidar as ld
import my_bag_utils as bu
import numpystream as ns
import rotation_detector as rd
import squeezedet as sd
import track
import video

from config import *
from nets import *
import train
import utils.util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '/home/eljefec/repo/squeezeDet/data/model_checkpoints/didi/model.ckpt-42000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'bag_dir', '/data/bags/didi-round2/release/car/testing', """ROS bag folder""")
tf.app.flags.DEFINE_string(
    'out_dir', 'NO_DEFAULT', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'didi', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'bag_file', '', """ROS bag.""")
# tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string(
    'do', 'NO_DEFAULT', """[video, tracker, print, tracklet].""")
tf.app.flags.DEFINE_boolean('include_car', False, """Whether to include car in tracklet.""")
tf.app.flags.DEFINE_boolean('include_ped', False, """Whether to include pedestrian in tracklet.""")

def generate_detections(bag_file, demo_net, skip_null, tracklet_file = None):
  """Detect image."""

  with sd.SqueezeDetector(demo_net) as det:

    mc = sd.get_model_config(demo_net)
    if demo_net == 'squeezeDet':
      input_generator = generate_camera_images(bag_file, mc, tracklet_file)
    elif demo_net == 'squeezeDet+':
      input_generator = generate_camera_images(bag_file, mc, tracklet_file)
    elif demo_net == 'didi':
      input_generator = generate_lidar_birdseye(bag_file, mc)

    im = None
    final_boxes = None
    final_probs = None
    final_class = None

    frame_count = 0

    for im, token in input_generator:
      if im is not None:
        (final_boxes, final_probs, final_class) = det.detect(im)
        if skip_null:
          yield im, final_boxes, final_probs, final_class, token
      if not skip_null:
        yield im, final_boxes, final_probs, final_class, token

      frame_count += 1
      if frame_count % 1000 == 0:
        print('Processed {} frames.'.format(frame_count))

def generate_camera_images(bag_file, mc, tracklet_file):
  camera_converter = cc.CameraConverter()
  im = None
  generator = ns.generate_numpystream(bag_file, tracklet_file)
  for numpydata in generator:
    im = numpydata.image
    obs = numpydata.obs
    if im is not None:
      im = sd.undistort_and_crop(im, camera_converter, mc)
    # Must yield item for each frame in generator.
    yield im, obs

def generate_lidar_birdseye(bag_file, mc):
  im = None
  generator = ns.generate_numpystream(bag_file, tracklet_file = None)
  for numpydata in generator:
    lidar = numpydata.lidar
    if lidar is not None:
      birdseye = ld.lidar_to_birdseye(lidar, ld.slice_config())

      im = ci.crop_image(birdseye,
                         (mc.IMAGE_WIDTH + 1, mc.IMAGE_HEIGHT + 1, 3),
                         (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 3))
    # Must yield item for each frame in generator.
    yield im, lidar

def get_filename(bag_file):
  base = os.path.basename(bag_file)
  split = os.path.splitext(base)
  return split[0]

class Detector:
  def __init__(self, demo_net):
    self.mc = sd.get_model_config(demo_net)
    self.rotation_detector = rd.get_latest_detector()

  def make_detection_video(self, bag_file, demo_net):
    cls2clr = {
        'car': (255, 191, 0),
        'cyclist': (0, 191, 255),
        'pedestrian':(255, 0, 191)
    }

    video_maker = video.VideoMaker(FLAGS.out_dir)
    generator = generate_detections(bag_file, demo_net = demo_net, skip_null = True)

    for im, boxes, probs, classes, lidar in generator:
      train._draw_box(
          im, boxes,
          [self.mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
              for idx, prob in zip(classes, probs)],
          cdict=cls2clr,
      )
      video_maker.add_image(im)

    video_filename = get_filename(bag_file) + '.mp4'
    video_maker.make_video(video_filename)

  def try_tracker(self, bag_file):
    car_tracker = track.Tracker(img_shape = (self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT),
                                  heatmap_window_size = 5,
                                  heatmap_threshold_per_frame = 0.5,
                                  vehicle_window_size = 5)

    ped_tracker = track.Tracker(img_shape = (self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT),
                                  heatmap_window_size = 5,
                                  heatmap_threshold_per_frame = 0.5,
                                  vehicle_window_size = 5)

    frame_idx = 0

    generator = generate_detections(bag_file, demo_net = 'didi', skip_null = True)
    for im, boxes, probs, classes, lidar in generator:
      frame_idx += 1

      car_boxes = []
      car_probs = []
      ped_boxes = []
      ped_probs = []

      for box, prob, class_idx in zip(boxes, probs, classes):
        box = utils.util.bbox_transform(box)
        if class_idx == 0:
          car_boxes.append(box)
          car_probs.append(prob)
        elif class_idx == 1:
          ped_boxes.append(box)
          ped_probs.append(prob)

      car_tracker.track(car_boxes, car_probs)
      ped_tracker.track(ped_boxes, ped_probs)

      print('Frame: {}, Car Boxes: {}, Ped Boxes: {} Tracked Cars: {}, Tracked Peds: {}'.format(frame_idx, len(car_boxes), len(ped_boxes), len(car_tracker.vehicles), len(ped_tracker.vehicles)))

  def print_detections(self, bag_file):
    generator = generate_detections(bag_file, demo_net = 'didi', skip_null = True)
    for im, boxes, probs, classes, lidar in generator:
      print('boxes', boxes)
      print('probs', probs)
      print('classes', classes)

  def gen_tracklet(self, bag_file, include_car, include_ped):

    def make_pose(x, y, rz):
      # Estimate tz from histogram.
      return {'tx': x,
              'ty': y,
              'tz': -0.9,
              'rx': 0,
              'ry': 0,
              'rz': rz}

    prev_car_pose = make_pose(0, 0, 0)
    prev_ped_pose = make_pose(0, 0, 0)
    predicted_yaw = 0

    # l, w, h from histogram
    car_tracklet = generate_tracklet.Tracklet(object_type='Car', l=4.3, w=1.7, h=1.7, first_frame=0)
    ped_tracklet = generate_tracklet.Tracklet(object_type='Pedestrian', l=0.8, w=0.8, h=1.708, first_frame=0)

    generator = generate_detections(bag_file, demo_net = 'didi', skip_null = False)
    for im, boxes, probs, classes, lidar in generator:
      car_found = False
      ped_found = False
      if (im is not None and boxes is not None
          and probs is not None and classes is not None):
        # Assume decreasing order of probability
        for box, prob, class_idx in zip(boxes, probs, classes):
          global_box = ld.birdseye_to_global(box, ld.slice_config())

          if lidar is not None:
            car_box = rd.get_birdseye_box(lidar, (global_box[0], global_box[1]))
            predicted_yaw = self.rotation_detector.detect_rotation(car_box)

          sd.correct_global(global_box, class_idx)

          pose = make_pose(global_box[0], global_box[1], predicted_yaw)

          if not car_found and class_idx == sd.CAR_CLASS:
            # box is in center form (cx, cy, w, h)
            car_tracklet.poses.append(pose)
            prev_car_pose = pose
            car_found = True
          if not ped_found and class_idx == sd.PED_CLASS:
            ped_tracklet.poses.append(pose)
            prev_ped_pose = pose
            ped_found = True

          if car_found and ped_found:
            break
      if not car_found:
        car_tracklet.poses.append(prev_car_pose)
      if not ped_found:
        ped_tracklet.poses.append(prev_ped_pose)

    tracklet_collection = generate_tracklet.TrackletCollection()
    if include_car:
      tracklet_collection.tracklets.append(car_tracklet)
    if include_ped:
      tracklet_collection.tracklets.append(ped_tracklet)

    tracklet_file = os.path.join(FLAGS.out_dir, get_filename(bag_file) + '.xml')

    tracklet_collection.write_xml(tracklet_file)

def process_bag(bag_file):
  detector = Detector(FLAGS.demo_net)

  print()
  if FLAGS.do == 'video':
    print('Making video')
    detector.make_detection_video(bag_file, FLAGS.demo_net)
  elif FLAGS.do == 'tracker':
    print('Trying tracker')
    detector.try_tracker(bag_file)
  elif FLAGS.do == 'print':
    print('Print detections')
    detector.print_detections(bag_file)
  elif FLAGS.do == 'tracklet':
    print('Generate tracklet')
    print('Include car: ', FLAGS.include_car)
    print('Include ped: ', FLAGS.include_ped)
    if not FLAGS.include_car and not FLAGS.include_ped:
      print('Must include one of the obstacle types.')
      exit()
    detector.gen_tracklet(bag_file, FLAGS.include_car, FLAGS.include_ped)
  else:
    raise ValueError('Unrecognized FLAGS.do: [{}]'.format(FLAGS.do))

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.bag_file:
    print('Processing single bag. {}'.format(FLAGS.bag_file))
    process_bag(FLAGS.bag_file)
  elif FLAGS.bag_dir:
    print('Processing bag folder. {}'.format(FLAGS.bag_dir))
    bags = bu.find_bags(FLAGS.bag_dir)
    for bag in bags:
      process_bag(bag)
  else:
    print('Neither bag_file nor bag_dir specified.')

if __name__ == '__main__':
    tf.app.run()
