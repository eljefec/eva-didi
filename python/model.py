import keras.layers
from keras.layers.core import Flatten
from keras.layers import Conv2D, Input, Dense
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
import traindata

IMAGE_SHAPE = (1096,1368,3)
PANORAMA_SHAPE = (38,901,1)
SLICES_SHAPE = (200,200,8)

INPUT_IMAGE = 'input_image'
INPUT_LIDAR_PANORAMA = 'input_lidar_panorama'
INPUT_LIDAR_SLICES = 'input_lidar_slices'
OUTPUT_POSE = 'output_pose'

def generate(batch_size, bag_file, tracklet_file):
    images = []
    panoramas = []
    slices_list = []
    poses = []

    while True:
        stream = traindata.TrainDataStream()
        datastream.start_read(bag_file, tracklet_file)
        while not datastream.empty():
            datum = datastream.next()

            if datum.lidar_panorama is None:
                panorama = np.zeros(PANORAMA_SHAPE, dtype = np.uint8)
            else:
                panorama = datum.lidar_panorama

            if datum.lidar_slices is None:
                slices = np.zeros(SLICES_SHAPE, dtype = np.uint8)
            else:
                slices = datum.lidar_slices

            images.append(datum.image)
            panoramas.append(np.expand_dims(panorama, axis=-1))
            slices_list.append(slices)
            poses.append(datum.pose)

            if batch_size == len(images):
                image_batch = np.stack(images)
                panorama_batch = np.stack(panoramas)
                slices_batch = np.stack(slices_list)
                pose_batch = np.stack(poses)

                images[:] = []
                panoramas[:] = []
                slices_list[:] = []
                poses[:] = []

                yield ({INPUT_IMAGE: image_batch,
                        INPUT_LIDAR_PANORAMA: panorama_batch,
                        INPUT_LIDAR_SLICES: slices_batch},
                       {OUTPUT_POSE: pose_batch}
                      )

def train_model():
    input_image = Input(shape = IMAGE_SHAPE,
                        dtype = 'float32',
                        name = INPUT_IMAGE)
    pool = AveragePooling2D()(input_image)
    pool = AveragePooling2D()(pool)
    pool = AveragePooling2D()(pool)
    pool = AveragePooling2D()(pool)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(pool)
    image_out = Flatten()(conv)
    # image_out = Dense(32, activation='relu')(conv)

    input_lidar_panorama = Input(shape = PANORAMA_SHAPE,
                                 dtype = 'float32',
                                 name = INPUT_LIDAR_PANORAMA)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(input_lidar_panorama)
    panorama_out = Flatten()(conv)

    input_lidar_slices = Input(shape = SLICES_SHAPE,
                               dtype = 'float32',
                               name = INPUT_LIDAR_SLICES)
    conv = Conv2D(32, kernel_size=3, strides=(2,2))(input_lidar_slices)
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
    print('model compiled.')
    print(model)
#    hist = model.fit_generator(generate(10,
#                                        '/data/Didi-Release-2/Data/1/2.bag',
#                                        '/data/output/test/2/tracklet_labels.xml'),
#                               steps_per_epoch=1584,
#                               epochs=10)
#    print(hist)

if __name__ == '__main__':
    train_model()
