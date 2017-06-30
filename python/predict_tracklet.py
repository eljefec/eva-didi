# Based loosely on https://github.com/BichenWuUCB/squeezeDet/blob/master/src/demo.py

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

import lidar as ld
import numpystream as ns
import track

from config import *
# from squeezeDet.src.train import _draw_box
from nets import *
import utils.util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', '/home/eljefec/repo/squeezeDet/data/model_checkpoints/didi/model.ckpt-6000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/KITTI/training/image_2/0*0000.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'didi', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'bag_file', '/data/bags/didi-round2/release/car/training/nissan_driving_past_it/nissan07.bag', """ROS bag.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def predict_tracklet(bag_file):
  """Detect image."""

  generator = ns.generate_numpystream(bag_file, tracklet_file = None)

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+' \
         or FLAGS.demo_net == 'didi', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'didi':
      mc = didi_squeezeDet_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    car_tracker = track.Tracker(img_shape = (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT),
                                heatmap_window_size = 5,
                                heatmap_threshold_per_frame = 0.5,
                                vehicle_window_size = 5)

    ped_tracker = track.Tracker(img_shape = (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT),
                                heatmap_window_size = 5,
                                heatmap_threshold_per_frame = 0.5,
                                vehicle_window_size = 5)

    frame_idx = 0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      # for f in glob.iglob(FLAGS.input_path):
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

          print('det_boxes.shape', det_boxes.shape)

          # Filter
          final_boxes, final_probs, final_class = model.filter_prediction(
              det_boxes[0], det_probs[0], det_class[0])

          keep_idx    = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
          final_boxes = [final_boxes[idx] for idx in keep_idx]
          final_probs = [final_probs[idx] for idx in keep_idx]
          final_class = [final_class[idx] for idx in keep_idx]

          if final_boxes:
            print('final_boxes', final_boxes)

          car_boxes = []
          car_probs = []
          ped_boxes = []
          ped_probs = []

          for box, prob, class_idx in zip(final_boxes, final_probs, final_class):
            box = utils.util.bbox_transform(box)
            if class_idx == 0:
              car_boxes.append(box)
              car_probs.append(prob)
            elif class_idx == 1:
              ped_boxes.append(box)
              ped_probs.append(prob)

          car_tracker.track(car_boxes, car_probs)
          ped_tracker.track(ped_boxes, ped_probs)

          print('Frame: {}, Cars: {}, Pedestrians: {}'.format(frame_idx, len(car_tracker.vehicles), len(ped_tracker.vehicles)))

        frame_idx += 1


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    predict_tracklet(FLAGS.bag_file)

if __name__ == '__main__':
    tf.app.run()
