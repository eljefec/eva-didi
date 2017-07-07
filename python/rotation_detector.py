from __future__ import print_function

import crop_images as ci
import lidar as ld
import multibag as mb
import numpy as np
import numpystream
import pickle
import os
import util.stopwatch
import cv2
import math
import random

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.layers
from keras.layers.core import Flatten, Dropout, Lambda
from keras.layers import Conv2D, Dense, Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
import tensorflow as tf

INPUT_SHAPE=(50,50,3)

MODEL_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'
HISTORY_DIR = 'history'

def build_model(dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = INPUT_SHAPE))
    model.add(Conv2D(3, (1, 1), activation='relu'))
    model.add(Conv2D(12, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    return model

def get_birdseye_box(lidar, obs_position):
    slice_config = ld.slice_config()
    slice_config.SIDE_RANGE = (-2.5, 2.5)
    slice_config.FWD_RANGE = (-2.5, 2.5)

    birdseye = ld.lidar_to_birdseye(lidar,
                                    slice_config,
                                    return_points = False,
                                    center = (obs_position[0], obs_position[1]))
    return ci.crop_image(birdseye, (51, 51, 3), INPUT_SHAPE)

def generate_birdseye_boxes_single(multi, infinite):
    generator = multi.generate(infinite)
    for numpydata in generator:
        lidar = numpydata.lidar
        obs = numpydata.obs[1]
        if lidar is not None:
            birdseye_box = get_birdseye_box(lidar, (obs.position[0], obs.position[1]))

            yield birdseye_box, obs.yaw

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_image_label_dirs(dir):
    imagedir = os.path.join(dir, 'image')
    labeldir = os.path.join(dir, 'label')

    return (imagedir, labeldir)

def get_image_path(imagedir, id):
    return util.traingen.get_example_path(imagedir, id, 'png')

def get_label_path(labeldir, id):
    return util.traingen.get_example_path(labeldir, id, 'txt')

def generate_training_data(multi, outdir):
    imagedir, labeldir = get_image_label_dirs(outdir)

    makedir(imagedir)
    makedir(labeldir)

    id = 0
    generator = generate_birdseye_boxes_single(multi, infinite = False)
    for birdseye_box, yaw in generator:
        image_path = get_image_path(imagedir, id)
        cv2.imwrite(image_path, birdseye_box)

        label_path = get_label_path(labeldir, id)
        with open(label_path, 'w') as f:
            print(yaw, file=f)

        id += 1

    util.traingen.write_train_val(outdir, id)

def rotate_image(img, radians):
    (rows, cols, channels) = img.shape
    degrees = math.degrees(radians)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def normalize_angle(radians):
    while radians > math.pi:
        radians -= math.pi
    while radians < -math.pi:
        radians += math.pi
    return radians

def augment_example(orig_img, orig_yaw):
    rotation_radians = random.uniform(-math.pi, math.pi)
    # Rotate image
    new_img = rotate_image(orig_img, rotation_radians)
    # Calculate new yaw
    new_yaw = orig_yaw + rotation_radians
    new_yaw = normalize_angle(new_yaw)
    return (new_img, new_yaw)

def try_rotating_images(train_dir):
    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')

    multi = mb.MultiBagStream(bt, numpystream.generate_numpystream)
    generator = generate_birdseye_boxes_single(multi, infinite = False)
    count = 0
    frames_since_last_conversion = 0
    for birdseye_box, yaw in generator:
        if (yaw > (math.pi / 4) or yaw < (-math.pi / 4)) and frames_since_last_conversion > 10:
            # Try to undo rotation with negative yaw.
            rotated = rotate_image(birdseye_box, -yaw)
            # Expect car to have zero rotation in image.
            cv2.imwrite('rotate_test_{}.png'.format(count), rotated)
            print('count: {}, orig_yaw: {}'.format(count, yaw))
            count += 1
            frames_since_last_conversion = 0
            if count % 10 == 0:
                return
        else:
            frames_since_last_conversion += 1

def generate_birdseye_boxes_dir(train_dir, index_file, augment, infinite = True):
    while True:
        ids = []
        with open(os.path.join(train_dir, index_file), 'r') as f:
            for id in f:
                ids.append(int(id))

        imagedir, labeldir = get_image_label_dirs(train_dir)

        for id in ids:
            image_path = get_image_path(imagedir, id)
            birdseye_box = cv2.imread(image_path)

            label_path = get_label_path(labeldir, id)
            with open(label_path, 'r') as f:
                yaw = float(f.readline())

            if augment:
                (birdseye_box, yaw) = augment_example(birdseye_box, yaw)

            yield birdseye_box, yaw

        if not infinite:
            return

def generate_birdseye_boxes(single_generator, batch_size):
    images = []
    yaws = []
    for birdseye_box, yaw in single_generator:
        images.append(birdseye_box)
        yaws.append(yaw)

        if batch_size == len(images):
            image_batch = np.stack(images)
            yaw_batch = np.stack(yaws)

            images[:] = []
            yaws[:] = []

            yield (image_batch, yaw_batch)

def train_rotation_detector(train_dir):
    batch_size = 128

    generator_validation = generate_birdseye_boxes(
                                generate_birdseye_boxes_dir(train_dir, 'val.txt', augment = False),
                                batch_size)

    generator_train = generate_birdseye_boxes(
                                generate_birdseye_boxes_dir(train_dir, 'train.txt', augment = True),
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
    return '{}/model_{}{}.{}'.format(directory, util.stopwatch.format_now(), suffix, ext)

class RotationDetector:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def detect_rotation(self, birdseye_box):
        with self.graph.as_default():
            prediction = self.model.predict(np.array([birdseye_box]), batch_size=1, verbose=0)
            return prediction

def get_latest_detector():
    abs_checkpoint_dir = '/home/eljefec/repo/eva-didi/python/checkpoints'
    model_name = 'model_2017-07-02_18h10m55e40-vl0.49.h5'
    model_path = os.path.join(abs_checkpoint_dir, model_name)
    return RotationDetector(model_path)

def make_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def try_detector():
    detector = get_latest_detector()

    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')
    multi = mb.MultiBagStream(bt, numpystream.generate_numpystream)
    generator = generate_birdseye_boxes_single(multi, infinite = False)
    for birdseye_box, yaw in generator:
        prediction = detector.detect_rotation(birdseye_box)
        print('gt_yaw: [{}], predicted_yaw: [{}]'.format(yaw, prediction))

if __name__ == '__main__':
    make_dir(MODEL_DIR)
    make_dir(CHECKPOINT_DIR)
    make_dir(HISTORY_DIR)

    try_detector()
    exit()

    # bagdir = '/data/bags/didi-round2/release/car/training/'
    # bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    # bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')

    # multi = mb.MultiBagStream(bt, numpystream.generate_numpystream)

    train_dir = '/data/rot_train'
    # generate_training_data(multi, train_dir)

    train_rotation_detector(train_dir)
