import crop_images as ci
import lidar as ld
import multibag as mb
import numpy as np
import numpystream
import util.stopwatch
import cv2

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.layers
from keras.layers.core import Flatten, Dropout, Lambda
from keras.layers import Conv2D, Dense, Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

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
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    return model

def get_birdseye_box(lidar, obs):
    slice_config = ld.slice_config()
    slice_config.SIDE_RANGE = (-2.5, 2.5)
    slice_config.FWD_RANGE = (-2.5, 2.5)

    birdseye = ld.lidar_to_birdseye(lidar,
                                    slice_config,
                                    return_points = False,
                                    center = (obs.position[0], obs.position[1]))
    return ci.crop_image(birdseye, (51, 51, 3), INPUT_SHAPE)

def generate_birdseye_boxes(multi, batch_size, include_pred):
    generator = multi.generate(infinite = True)
    images = []
    rotations = []
    count = 0
    for numpydata in generator:
        lidar = numpydata.lidar
        obs = numpydata.obs[1]
        if lidar is not None:
            birdseye_box = get_birdseye_box(lidar, obs)

            if include_pred(count):
                images.append(birdseye_box)
                rotations.append(obs.yaw)

            count += 1

            if batch_size == len(images):
                image_batch = np.stack(images)
                rotation_batch = np.stack(rotations)

                images[:] = []
                rotations[:] = []

                yield (image_batch, rotation_batch)

def train_rotation_detector(multi):
    batch_size=32

    pred_validation = lambda count : count % 5 == 0
    pred_train = lambda count : count % 5 != 0

    generator_validation = generate_birdseye_boxes(multi, batch_size, pred_validation)
    generator_train = generate_birdseye_boxes(multi, batch_size, pred_train)

    checkpoint_path = get_model_filename(CHECKPOINT_DIR, suffix = 'e{epoch:02d}-vl{val_loss:.2f}')

    # Set up callbacks. Stop early if the model does not improve. Save model checkpoints.
    # Source: http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, verbose=0),
    ]

    model = build_model(dropout = 0.2)

    model.summary()

    # Fix error with TF and Keras
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

    hist = model.fit_generator(generator_train,
                               steps_per_epoch = int(0.8 * (multi.count() / batch_size)),
                               epochs = 20,
                               # Values for quick testing:
                               # steps_per_epoch = (128 / batch_size),
                               # epochs = 2,
                               validation_data = generator_validation,
                               validation_steps = int(0.2 * (multi.count() / batch_size)),
                               callbacks = callbacks)
    model.save(get_model_filename(MODEL_DIR))

    with open(get_model_filename(HISTORY_DIR, '', 'p'), 'wb') as f:
        pickle.dump(hist.history, f)

from util import *
def get_model_filename(directory, suffix = '', ext = 'h5'):
    return '{}/model_{}{}.{}'.format(directory, util.stopwatch.format_now(), suffix, ext)

def detect_rotation(birdseye):
    pass

def make_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    make_dir(MODEL_DIR)
    make_dir(CHECKPOINT_DIR)
    make_dir(HISTORY_DIR)

    bagdir = '/data/bags/didi-round2/release/car/training/suburu_leading_front_left'
    # bagdir = '/data/bags/didi-round2/release/car/training/'
    bt = mb.find_bag_tracklets(bagdir, '/data/tracklets')

    multi = mb.MultiBagStream(bt, numpystream.generate_numpystream)

    train_rotation_detector(multi)
