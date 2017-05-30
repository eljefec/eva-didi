from generator import *
import keras.layers
from keras.layers.core import Flatten
from keras.layers import Conv2D, Input, Dense
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
import numpy as np
import traindata

def build_model():
    input_image = Input(shape = IMAGE_SHAPE,
                        dtype = 'float32',
                        name = INPUT_IMAGE)
    pool = AveragePooling2D()(input_image)
    pool = AveragePooling2D()(pool)
    pool = AveragePooling2D()(pool)
    pool = AveragePooling2D()(pool)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(pool)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(conv)
    image_out = Flatten()(conv)
    # image_out = Dense(32, activation='relu')(conv)

    input_lidar_panorama = Input(shape = PANORAMA_SHAPE,
                                 dtype = 'float32',
                                 name = INPUT_LIDAR_PANORAMA)
    pool = AveragePooling2D()(input_lidar_panorama)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(pool)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(conv)
    panorama_out = Flatten()(conv)

    input_lidar_slices = Input(shape = SLICES_SHAPE,
                               dtype = 'float32',
                               name = INPUT_LIDAR_SLICES)
    pool = AveragePooling2D()(input_lidar_slices)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(pool)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(conv)
    slices_out = Flatten()(conv)

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

MODEL_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'

def train_model(model):
    batch_size = 64
    generator = TrainDataGenerator('/data/Didi-Release-2/Data/', '/data/output/tracklet/')
    hist = model.fit_generator(generator.generate(batch_size),
                               # steps_per_epoch = (generator.get_count() / batch_size),
                               steps_per_epoch = (1000 / batch_size),
                               epochs = 1)
    model.save(get_model_filename(MODEL_DIR))
    print(hist)

import stopwatch
def get_model_filename(directory, suffix = ''):
    return '{}/model_{}{}.h5'.format(directory, stopwatch.format_now(), suffix)

def make_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    make_dir(MODEL_DIR)
    make_dir(CHECKPOINT_DIR)

    model = build_model()
    model.summary()
    train_model(model)
    # TODO: Save checkpoints.
