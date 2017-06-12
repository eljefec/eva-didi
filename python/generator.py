import numpy as np

IMAGE_SHAPE = (1096,1368,3)
PANORAMA_SHAPE = (38,901,1)
SLICES_SHAPE = (300,300,4,1)

INPUT_IMAGE = 'input_image'
INPUT_LIDAR_PANORAMA = 'input_lidar_panorama'
INPUT_LIDAR_SLICES = 'input_lidar_slices'
OUTPUT_POSE = 'output_pose'

class DatumChecker:
    def __init__(self):
        self.null_datum_count = 0

    def report_datum(self, datum):
        if datum is None:
            self.null_datum_count += 1
            if (self.null_datum_count > 100):
                print('Warning: Datastream returned many null messages in a row.')

class TrainDataGenerator:
    def __init__(self, multibag, include_ground_truth):
        self.multibag = multibag
        self.include_ground_truth = include_ground_truth

    def get_count(self):
        return self.multibag.count()

    def generate(self, batch_size):
        generator = self.multibag.generate()

        images = []
        panoramas = []
        slices_list = []
        if self.include_ground_truth:
            poses = []

        while True:
            datum_checker = DatumChecker()
            datum = None
            while datum is None:
                datum = next(generator)
                datum_checker.report_datum(datum)

            if datum.lidar_panorama is None:
                panorama = np.zeros(PANORAMA_SHAPE, dtype = np.uint8)
            else:
                panorama = np.expand_dims(datum.lidar_panorama, axis=-1)

            if datum.lidar_slices is None:
                slices = np.zeros(SLICES_SHAPE, dtype = np.uint8)
            else:
                slices = np.expand_dims(datum.lidar_slices, axis=-1)

            #if np.expand_dims(panorama, axis=-1).shape != PANORAMA_SHAPE:
            #    print("Panorama shape {0} doesn't match required shape {1}.".format(panorama.shape, PANORAMA_SHAPE))
            #    continue

            images.append(datum.image)
            panoramas.append(panorama)
            slices_list.append(slices)
            if self.include_ground_truth:
                poses.append(datum.pose)

            if (batch_size == len(images)):
                image_batch = np.stack(images)
                panorama_batch = np.stack(panoramas)
                slices_batch = np.stack(slices_list)
                if self.include_ground_truth:
                    pose_batch = np.stack(poses)

                images[:] = []
                panoramas[:] = []
                slices_list[:] = []
                if self.include_ground_truth:
                    poses[:] = []

                if self.include_ground_truth:
                    yield ({INPUT_IMAGE: image_batch,
                            INPUT_LIDAR_PANORAMA: panorama_batch,
                            INPUT_LIDAR_SLICES: slices_batch},
                           {OUTPUT_POSE: pose_batch}
                          )
                else:
                    yield {INPUT_IMAGE: image_batch,
                           INPUT_LIDAR_PANORAMA: panorama_batch,
                           INPUT_LIDAR_SLICES: slices_batch}
