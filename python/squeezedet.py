# Based loosely on https://github.com/BichenWuUCB/squeezeDet/blob/master/src/demo.py

from __future__ import print_function

import numpy as np
import tensorflow as tf

from config import *
from nets import *
import camera_converter as cc
import train
import utils.util

CAR_CLASS = 0
PED_CLASS = 1

def undistort_and_crop(im, camera_converter, mc):
  im = camera_converter.undistort_image(im)
  width_start = int((im.shape[1] - mc.IMAGE_WIDTH) / 2)
  height_start = (800 - mc.IMAGE_HEIGHT)
  return im[height_start : height_start + mc.IMAGE_HEIGHT,
            width_start : width_start + mc.IMAGE_WIDTH,
            :]

def get_model_config(demo_net):
  assert demo_net == 'squeezeDet' or demo_net == 'squeezeDet+' \
         or demo_net == 'didi', \
      'Selected neural net architecture not supported: {}'.format(demo_net)

  if demo_net == 'squeezeDet':
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
  elif demo_net == 'squeezeDet+':
    mc = kitti_squeezeDetPlus_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
  elif demo_net == 'didi':
    mc = didi_squeezeDet_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
  return mc

class SqueezeDetector:
  def __init__(self, demo_net):
    assert demo_net == 'squeezeDet' or demo_net == 'squeezeDet+' \
           or demo_net == 'didi', \
        'Selected neural net architecture not supported: {}'.format(demo_net)

    self.demo_net = demo_net
    self.mc = None
    self.model = None
    self.sess = None
    self._prepare_graph()
    self.camera_converter = cc.CameraConverter()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.sess.close()

  def _prepare_graph(self):
    gpu = 0

    graph = tf.Graph()
    with graph.as_default():
      self.mc = get_model_config(self.demo_net)
      if self.demo_net == 'squeezeDet':
        self.model = SqueezeDet(self.mc, gpu)
        checkpoint = '/home/eljefec/repo/squeezeDet/data/model_checkpoints/squeezeDet/model.ckpt-87000'
      elif self.demo_net == 'squeezeDet+':
        self.model = SqueezeDetPlus(self.mc, gpu)
        checkpoint = '/home/eljefec/repo/squeezeDet/data/model_checkpoints/squeezeDetPlus/model.ckpt-95000'
      elif self.demo_net == 'didi':
        self.model = SqueezeDet(self.mc, gpu)
        checkpoint = '/home/eljefec/repo/squeezeDet/data/model_checkpoints/didi/model.ckpt-42000'

      saver = tf.train.Saver(self.model.model_params)
      self.sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
      saver.restore(self.sess, checkpoint)

  def detect(self, im):
    im = im.astype(np.float32, copy=False)
    input_image = im - self.mc.BGR_MEANS

    # Detect
    det_boxes, det_probs, det_class = self.sess.run(
        [self.model.det_boxes, self.model.det_probs, self.model.det_class],
        feed_dict={self.model.image_input:[input_image]})

    # Filter
    final_boxes, final_probs, final_class = self.model.filter_prediction(
        det_boxes[0], det_probs[0], det_class[0])

    keep_idx    = [idx for idx in range(len(final_probs)) \
                      if final_probs[idx] > self.mc.PLOT_PROB_THRESH]
    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    return final_boxes, final_probs, final_class

  def undistort_and_crop(self, im):
    return undistort_and_crop(im, self.camera_converter, self.mc)

def correct_global(global_box, class_idx):
  # Slight correction needed due to cropping of birds eye image.
  if class_idx == CAR_CLASS:
    global_box[0] += 0.365754455
    global_box[1] += 0.368273374
  elif class_idx == PED_CLASS:
    global_box[0] += 0.191718419
    global_box[1] += 0.154991700
