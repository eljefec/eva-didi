import cv2
import numpy as np
import os
import random
import yaml

import camera_converter as cc
import multibag as mb
import numpystream as ns
import squeezedet
import util.traingen

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.layers
from keras.layers.core import Flatten, Dropout, Lambda
from keras.layers import Conv2D, Dense, Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

INPUT_SHAPE=(6,)

CAMERA_ROOT = '/home/eljefec/repo/eva-didi/camera_det'
MODEL_DIR = os.path.join(CAMERA_ROOT, 'models')
CHECKPOINT_DIR = os.path.join(CAMERA_ROOT, 'checkpoints')
HISTORY_DIR = os.path.join(CAMERA_ROOT, 'history')

class ImageBoxToPosePredictor:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict_pose(self, image_box):
        prediction = self.model.predict(np.array([image_box]), batch_size=1, verbose=0)
        return prediction

class CameraDetector:
    def __init__(self):
        self.squeezedet = squeezedet.SqueezeDetector(demo_net = 'squeezeDet')
        self.box_to_pose_predictor = get_latest_predictor()

    def detect_obstacles(self, image):
        boxes, probs, classes = self.squeezedet.detect(image)
        found_car = None
        found_ped = None
        for box, prob, class_idx in zip(boxes, probs, classes):
            pose = self.box_to_pose_predictor.predict_pose(box)

            # TODO: Check if correction is needed.
            correct_global(pose, class_idx)

            if found_car is None and class_idx == CAR_CLASS:
                found_car = pose

            if found_ped is None and class_idx == PED_CLASS:
                found_ped = pose

            if found_car is not None and found_ped is not None:
                break
        return (found_car, found_ped)

def get_latest_predictor():
    # model_name = 'model_2017-07-02_18h10m55e40-vl0.49.h5'
    model_path = os.path.join(CHECKPOINT_DIR, model_name)
    return ImageBoxToPosePredictor(model_path)

def build_model(dropout):
    model = Sequential()
    model.add(BatchNormalization(input_shape = INPUT_SHAPE))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4))

    return model

def try_undistort(desired_count):
    undist = cc.CameraConverter()

    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_at_distance'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, ns.generate_numpystream)
    generator = multi.generate(infinite = False)
    count = 0
    output_count = 0
    for numpydata in generator:
        im = numpydata.image
        frame_idx, obs = numpydata.obs
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        undistorted = undist.undistort_image(im)
        if count % 25 == 0:
            cv2.imwrite('/data/dev/camera/orig_{}.png'.format(count), im)

            # Print center.
            img_point = undist.project_point(obs.position)
            cv2.circle(undistorted, (int(img_point[0]), int(img_point[1])), radius = 5, color = (255, 0, 0), thickness=2)

            # Print bbox corners.
            img_points = undist.project_points(obs.get_bbox().transpose())
            for img_point in img_points:
                cv2.circle(undistorted, (int(img_point[0]), int(img_point[1])), radius = 5, color = (0, 255, 0), thickness=2)

            cv2.imwrite('/data/dev/camera/undist_{}.png'.format(count), undistorted)
            output_count += 1
        count += 1
        if desired_count is not None and output_count == desired_count:
            return

def generate_top_boxes(bag_file, tracklet_file):
    generator = squeezedet.generate_detections(bag_file, demo_net = 'squeezeDet', skip_null = True, tracklet_file = tracklet_file)
    for im, boxes, probs, classes, obs in generator:
      car_found = False
      ped_found = False
      top_car = (None, None, None)
      top_ped = (None, None, None)

      if (im is not None and boxes is not None
          and probs is not None and classes is not None):
        # Assume decreasing order of probability
        for box, prob, class_idx in zip(boxes, probs, classes):
          if not car_found and class_idx == squeezedet.CAR_CLASS:
            # box is in center form (cx, cy, w, h)
            top_car = (box, prob, class_idx)
            car_found = True
          if not ped_found and class_idx == squeezedet.PED_CLASS:
            top_ped = (box, prob, class_idx)
            ped_found = True

          if car_found and ped_found:
            break

      yield (top_car, top_ped, obs, im)

def generate_training_data(bag_file, tracklet_file):
    mc = squeezedet.get_model_config(demo_net = 'squeezeDet')
    camera_converter = cc.CameraConverter()

    generator = generate_top_boxes(bag_file, tracklet_file)
    for top_car, top_ped, (frame_idx, obs), im in generator:
        top_obs = None
        if obs.object_type == 'Car' and top_car is not None:
            top_obs = top_car
        elif obs.object_type == 'Pedestrian' and top_ped is not None:
            top_obs = top_ped

        if top_obs is not None:
            (box, prob, class_idx) = top_obs

            if (box is not None and
                camera_converter.obstacle_is_in_view(obs)):

                yield (np.array([box[0], box[1], box[2], box[3], prob, class_idx]),
                       np.array([obs.position[0], obs.position[1], obs.position[2], obs.yaw]),
                       im)

def augment_example_unbounded(bbox, label, camera_converter):
    orig_obj_point = np.array([label[0], label[1], label[2]])
    orig_obj_img_point = camera_converter.project_point(orig_obj_point)

    obj_horizontal_shift = random.uniform(-0.5, 0.5)
    new_obj_point = orig_obj_point - np.array([0, -obj_horizontal_shift, 0])
    new_obj_img_point = camera_converter.project_point(new_obj_point)

    img_horizontal_shift = new_obj_img_point[0] - orig_obj_img_point[0]

    return (bbox + np.array([img_horizontal_shift, 0, 0, 0, 0, 0]),
            label + np.array([0, -obj_horizontal_shift, 0, 0]))

def augment_example(bbox, label, camera_converter):
    augmented = augment_example_unbounded(bbox, label, camera_converter)
    if camera_converter.bbox_is_in_view(augmented[0]):
        return augmented
    else:
        return bbox, label

def generate_training_data_multi(bag_tracklets):
    for bt in bag_tracklets:
        generator = generate_training_data(bt.bag, bt.tracklet)
        for example in generator:
            yield example

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_bbox_label_dirs(dir):
    bboxdir = os.path.join(dir, 'bbox')
    labeldir = os.path.join(dir, 'label')
    imagedir = os.path.join(dir, 'image')

    return (bboxdir, labeldir, imagedir)

def get_bbox_path(bboxdir, id):
    return util.traingen.get_example_path(bboxdir, id, 'txt')

def get_label_path(labeldir, id):
    return util.traingen.get_example_path(labeldir, id, 'txt')

def get_image_path(imagedir, id):
    return util.traingen.get_example_path(imagedir, id, 'png')

def write_training_data(bag_tracklets, outdir):
    bboxdir, labeldir, imagedir = get_bbox_label_dirs(outdir)

    makedir(bboxdir)
    makedir(labeldir)
    makedir(imagedir)

    generator = generate_training_data_multi(bag_tracklets)
    id = 0
    for bbox, label, im in generator:
        bbox_path = get_bbox_path(bboxdir, id)
        np.savetxt(bbox_path, bbox)

        label_path = get_label_path(labeldir, id)
        np.savetxt(label_path, label)

        # image_path = get_image_path(imagedir, id)
        # cv2.imwrite(image_path, im)
        id += 1

        if id % 1000 == 0:
            print('Wrote {} examples.'.format(id))

    print('Finished. Wrote {} examples.'.format(id))

    util.traingen.write_train_val(id)

def generate_camera_boxes_dir(train_dir, index_file, augment, infinite = True):
    camera_converter = cc.CameraConverter()
    while infinite:
        ids = []
        with open(os.path.join(train_dir, index_file), 'r') as f:
            for id in f:
                ids.append(int(id))

        bboxdir, labeldir, imagedir = get_bbox_label_dirs(train_dir)

        for id in ids:
            bbox_path = get_bbox_path(bboxdir, id)
            bbox = np.loadtxt(bbox_path)

            label_path = get_label_path(labeldir, id)
            label = np.loadtxt(label_path)

            if augment:
                (bbox, label) = augment_example(bbox, label, camera_converter)

            yield bbox, label

def generate_batches(single_generator, batch_size):
    bboxes = []
    labels = []
    for bbox, label in single_generator:
        bboxes.append(bbox)
        labels.append(label)

        if batch_size == len(bboxes):
            bbox_batch = np.stack(bboxes)
            label_batch = np.stack(labels)

            bboxes[:] = []
            labels[:] = []

            yield (bbox_batch, label_batch)

def train_detector(train_dir):
    batch_size = 128

    generator_validation = generate_batches(
                                generate_camera_boxes_dir(train_dir, 'val.txt', augment = False),
                                batch_size)

    generator_train = generate_batches(
                                generate_camera_boxes_dir(train_dir, 'train.txt', augment = True),
                                batch_size)

    checkpoint_path = get_model_filename(CHECKPOINT_DIR, suffix = 'e{epoch:02d}-vl{val_loss:.2f}')

    # Set up callbacks. Stop early if the model does not improve. Save model checkpoints.
    # Source: http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, verbose=0),
    ]

    model = build_model(dropout = 0.4)

    model.summary()

    # Fix error with TF and Keras
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse')

    steps_per_epoch = util.traingen.get_size(train_dir, 'train.txt') / batch_size
    validation_steps = util.traingen.get_size(train_dir, 'val.txt') / batch_size

    hist = model.fit_generator(generator_train,
                               steps_per_epoch = steps_per_epoch,
                               epochs = 200,
                               # Values for quick testing:
                               # steps_per_epoch = (128 / batch_size),
                               # epochs = 2,
                               validation_data = generator_validation,
                               validation_steps = validation_steps,
                               callbacks = callbacks)
    model.save(get_model_filename(MODEL_DIR))

    with open(get_model_filename(HISTORY_DIR, '', 'p'), 'wb') as f:
        pickle.dump(hist.history, f)

from util import *
def get_model_filename(directory, suffix = '', ext = 'h5'):
    model_filename = 'model_{}{}.{}'.format(util.stopwatch.format_now(), suffix, ext)
    return os.path.join(directory, model_filename)

def try_augmenting_camera_boxes():
    camera_converter = cc.CameraConverter()
    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    generator = generate_training_data_multi(bt)
    count = 0
    for bbox, label, im in generator:
        new_bbox, new_label = augment_example(bbox, label, camera_converter)
        print('bbox', bbox)
        print('new_bbox', new_bbox)
        print('label', label)
        print('new_label', new_label)
        count += 1
        if count == 10:
            return

def make_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    make_dir(MODEL_DIR)
    make_dir(CHECKPOINT_DIR)
    make_dir(HISTORY_DIR)

    train_dir = '/home/eljefec/repo/squeezeDet/data/KITTI/camera_train'
    util.traingen.write_train_val(train_dir, 29026)
    # try_augmenting_camera_boxes()

    # bag_tracklets = mb.find_bag_tracklets('/data/bags/', '/data/tracklets')
    # write_training_data(bag_tracklets, train_dir)
    train_detector(train_dir)

    exit()

    import os
    path = '/data/dev/camera'
    if not os.path.exists(path):
        os.makedirs('/data/dev/camera')
    try_undistort(None)
