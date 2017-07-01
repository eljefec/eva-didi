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

import generate_tracklet
import lidar as ld
import my_bag_utils as bu
import numpystream as ns
import track
import video

from config import *
from nets import *
import train
import utils.util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '/home/eljefec/repo/squeezeDet/data/model_checkpoints/didi/model.ckpt-6000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'bag_dir', '/data/bags/didi-round2/release/car/testing', """ROS bag folder""")
tf.app.flags.DEFINE_string(
    'out_dir', '/data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'didi', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'bag_file', '', """ROS bag.""")
# tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string(
    'do', 'video', """[video, tracker, print, tracklet].""")
tf.app.flags.DEFINE_boolean('include_car', False, """Whether to include car in tracklet.""")
tf.app.flags.DEFINE_boolean('include_ped', False, """Whether to include pedestrian in tracklet.""")

def generate_obstacle_detections(bag_file, mc, skip_null = True):
  """Detect image."""

  generator = ns.generate_numpystream(bag_file, tracklet_file = None)

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+' \
         or FLAGS.demo_net == 'didi', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    if FLAGS.demo_net == 'squeezeDet':
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      model = SqueezeDetPlus(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'didi':
      model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      im = None
      final_boxes = None
      final_probs = None
      final_class = None

      for numpydata in generator:
        lidar = numpydata.lidar
        if lidar is not None:
          lidar = ld.lidar_to_birdseye(lidar)

          im = cv2.resize(lidar, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
          im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
          im = im.astype(np.float32, copy=False)
          input_image = im - mc.BGR_MEANS

          # Detect
          det_boxes, det_probs, det_class = sess.run(
              [model.det_boxes, model.det_probs, model.det_class],
              feed_dict={model.image_input:[input_image]})

          # Filter
          final_boxes, final_probs, final_class = model.filter_prediction(
              det_boxes[0], det_probs[0], det_class[0])

          keep_idx    = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
          final_boxes = [final_boxes[idx] for idx in keep_idx]
          final_probs = [final_probs[idx] for idx in keep_idx]
          final_class = [final_class[idx] for idx in keep_idx]

          if skip_null:
            yield im, final_boxes, final_probs, final_class
        if not skip_null:
          yield im, final_boxes, final_probs, final_class

def get_filename(bag_file):
  base = os.path.basename(bag_file)
  split = os.path.splitext(base)
  return split[0]

class Detector:
  def __init__(self, mc):
    self.mc = mc

  def make_detection_video(self, bag_file):
    cls2clr = {
        'car': (255, 191, 0),
        'cyclist': (0, 191, 255),
        'pedestrian':(255, 0, 191)
    }

    video_maker = video.VideoMaker(FLAGS.out_dir)
    generator = generate_obstacle_detections(bag_file, self.mc)

    for im, boxes, probs, classes in generator:
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

    generator = generate_obstacle_detections(bag_file, self.mc)
    for im, boxes, probs, classes in generator:
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
    generator = generate_obstacle_detections(bag_file, self.mc)
    for im, boxes, probs, classes in generator:
      print('boxes', boxes)
      print('probs', probs)
      print('classes', classes)

  def gen_tracklet(self, bag_file, include_car, include_ped):

    def make_pose(x, y):
      # Estimate tz from histogram.
      return {'tx': x,
              'ty': y,
              'tz': -0.9,
              'rx': 0,
              'ry': 0,
              'rz': 0}

    CAR_CLASS = 0
    PED_CLASS = 1
    prev_car_pose = make_pose(0, 0)
    prev_ped_pose = make_pose(0, 0)

    # l, w, h from histogram
    car_tracklet = generate_tracklet.Tracklet(object_type='Car', l=4.3, w=1.7, h=1.7, first_frame=0)
    ped_tracklet = generate_tracklet.Tracklet(object_type='Pedestrian', l=0.8, w=0.8, h=1.7, first_frame=0)

    generator = generate_obstacle_detections(bag_file, self.mc)
    for im, boxes, probs, classes in generator:
      car_found = False
      ped_found = False
      if (im is not None and boxes is not None
          and probs is not None and classes is not None):
        # Assume decreasing order of probability
        for box, prob, class_idx in zip(boxes, probs, classes):
          global_box = ld.birdseye_to_global(box)
          pose = make_pose(global_box[0], global_box[1])

          if not car_found and class_idx == CAR_CLASS:
            # box is in center form (cx, cy, w, h)
            car_tracklet.poses.append(pose)
            prev_car_pose = pose
            car_found = True
          if not ped_found and class_idx == PED_CLASS:
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
  # Load model config
  if FLAGS.demo_net == 'squeezeDet':
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
  elif FLAGS.demo_net == 'squeezeDet+':
    mc = kitti_squeezeDetPlus_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
  elif FLAGS.demo_net == 'didi':
    mc = didi_squeezeDet_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False

  detector = Detector(mc)

  if FLAGS.do == 'video':
    print('Making video')
    detector.make_detection_video(bag_file)
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
    print('Nothing to do.')

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
