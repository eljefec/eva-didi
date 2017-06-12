import debug
from generator import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.layers
from keras.layers.core import Flatten, Dropout
from keras.layers import Conv2D, Conv3D, Input, Dense
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.models import Model
import multibag
import numpy as np
import pickle
import picklebag
import traindata

def pool_and_conv(x):
    x = MaxPooling2D()(x)
    x = Conv2D(32, kernel_size=3, strides=(2,2))(x)
    return x

def build_model(dropout_rate = 0.2):
    input_image = Input(shape = IMAGE_SHAPE,
                        dtype = 'float32',
                        name = INPUT_IMAGE)
    x = MaxPooling2D()(input_image)
    x = MaxPooling2D()(x)
    x = MaxPooling2D()(x)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(32, kernel_size=3, strides=(2,2))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, kernel_size=3, strides=(2,2))(x)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    image_out = Flatten()(x)
    # image_out = Dense(32, activation='relu')(conv)

    input_lidar_panorama = Input(shape = PANORAMA_SHAPE,
                                 dtype = 'float32',
                                 name = INPUT_LIDAR_PANORAMA)
    x = pool_and_conv(input_lidar_panorama)
    x = pool_and_conv(x)
    x = Dropout(dropout_rate)(x)
    panorama_out = Flatten()(x)

    input_lidar_slices = Input(shape = SLICES_SHAPE,
                               dtype = 'float32',
                               name = INPUT_LIDAR_SLICES)
    x = MaxPooling3D(pool_size=(2,2,1))(input_lidar_slices)
    x = Conv3D(32, kernel_size=3, strides=(2,2,1))(x)
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Dropout(dropout_rate)(x)
    x = Conv3D(32, kernel_size=2, strides=(2,2,1))(x)
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Dropout(dropout_rate)(x)
    slices_out = Flatten()(x)

    x = keras.layers.concatenate([image_out, panorama_out, slices_out])

    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    pose_output = Dense(9, name=OUTPUT_POSE)(x)

    model = Model(inputs=[input_image, input_lidar_panorama, input_lidar_slices],
                  outputs=[pose_output])

    # Fix error with TF and Keras
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

LOG_DIR = 'logs'
MODEL_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'
HISTORY_DIR = 'history'

def train_model(model):
    validation_batch_size = 128
    train_batch_size = 128
    bag_tracklets = multibag.find_bag_tracklets('/data/didi/didi-round1/Didi-Release-2/Data/', '/old_data/output/tracklet/')

    # Good shuffle seeds: (7, 0.15)
    shuffleseed = 7
    multibag.shuffle(bag_tracklets, shuffleseed)
    split = multibag.train_validation_split(bag_tracklets, 0.15)

    validation_stream = multibag.MultiBagStream(split.validation_bags)
    validation_generator = TrainDataGenerator(validation_stream,
                                              include_ground_truth = True)

    training_stream = multibag.MultiBagStream(split.train_bags)
    training_generator = TrainDataGenerator(training_stream,
                                            include_ground_truth = True)

    print('train: ', training_generator.get_count(), ', validation: ', validation_generator.get_count())

    checkpoint_path = get_model_filename(CHECKPOINT_DIR, suffix = 'e{epoch:02d}-vl{val_loss:.2f}')

    # Set up callbacks. Stop early if the model does not improve. Save model checkpoints.
    # Source: http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, verbose=0),
    ]

    hist = model.fit_generator(training_generator.generate(train_batch_size),
                               steps_per_epoch = (training_generator.get_count() / train_batch_size),
                               epochs = 100,
                               # Values for quick testing:
                               # steps_per_epoch = (128 / batch_size),
                               # epochs = 2,
                               validation_data = validation_generator.generate(validation_batch_size),
                               validation_steps = (validation_generator.get_count() / validation_batch_size),
                               callbacks = callbacks)
    model.save(get_model_filename(MODEL_DIR))
    # print(hist)

    with open(get_model_filename(HISTORY_DIR, '', 'p'), 'wb') as f:
        pickle.dump(hist.history, f)

import stopwatch
def get_model_filename(directory, suffix = '', ext = 'h5'):
    return '{}/model_{}{}.{}'.format(directory, stopwatch.format_now(), suffix, ext)

def make_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    make_dir(LOG_DIR)

    debugger = debug.Debugger(LOG_DIR + '/debugger.log', 20 * 60)

    make_dir(MODEL_DIR)
    make_dir(CHECKPOINT_DIR)
    make_dir(HISTORY_DIR)

    model = build_model()
    model.summary()
    train_model(model)

    debugger.shutdown()
