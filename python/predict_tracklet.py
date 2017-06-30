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

import lidar as ld
import numpystream as ns
import track

from config import *
from nets import *
import train
import utils.util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '/home/eljefec/repo/squeezeDet/data/model_checkpoints/didi/model.ckpt-6000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/KITTI/training/image_2/0*0000.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', '/data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'didi', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'bag_file', '/data/bags/didi-round2/release/car/training/nissan_driving_past_it/nissan07.bag', """ROS bag.""")
# tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_boolean('make_video', False, """Whether to make video.""")


def predict_tracklet(bag_file, make_video):
  """Detect image."""

  cls2clr = {
      'car': (255, 191, 0),
      'cyclist': (0, 191, 255),
      'pedestrian':(255, 0, 191)
  }

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

    if make_video:
      video_maker = VideoMaker()
    frame_idx = 0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      # for f in glob.iglob(FLAGS.input_path):
      for numpydata in generator:
        frame_idx += 1
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

          if make_video:
            train._draw_box(
                im, final_boxes,
                [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                    for idx, prob in zip(final_class, final_probs)],
                cdict=cls2clr,
            )
            video_maker.add_image(im)

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

          print('Frame: {}, Car Boxes: {}, Ped Boxes: {} Tracked Cars: {}, Tracked Peds: {}'.format(frame_idx, len(car_boxes), len(ped_boxes), len(car_tracker.vehicles), len(ped_tracker.vehicles)))

  if make_video:
    video_maker.make_video(filename = 'video_track.mp4')

class VideoMaker:
  def __init__(self):
    self.subclip_paths = []
    self.images = []
    self.frame_idx = 0

  def add_image(self, image):
    self.images.append(image)
    self.frame_idx += 1
    if self.frame_idx % 500 == 0:
      self._make_subclip()

  def _make_subclip(self):
    subclip_path = os.path.join(FLAGS.out_dir, 'subclip_{}.mp4'.format(self.frame_idx))
    clip = mpy.ImageSequenceClip(self.images, fps=20)
    clip.write_videofile(subclip_path)
    self.subclip_paths.append(subclip_path)
    del self.images[:]

  def make_video(self, filename):
    self._make_subclip()
    clips = []
    for subclip_path in self.subclip_paths:
      clips.append(mpy.VideoFileClip(subclip_path))
    final_clip = mpy.concatenate_videoclips(clips)
    final_clip.write_videofile(os.path.join(FLAGS.out_dir, filename))
    for subclip_path in self.subclip_paths:
      os.remove(subclip_path)

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  predict_tracklet(FLAGS.bag_file, FLAGS.make_video)

if __name__ == '__main__':
    tf.app.run()
