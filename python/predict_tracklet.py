import numpy as np
import os
import sensor_msgs.point_cloud2 as pc2
import tensorflow as tf

import detection_pipeline as dp
import image
import generate_tracklet
import my_bag_utils as bu
import sensor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'bag_dir', '/data/bags/didi-round2/release/car/testing', """ROS bag folder""")
tf.app.flags.DEFINE_string(
    'out_dir', 'NO_DEFAULT', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'bag_file', '', """ROS bag.""")
tf.app.flags.DEFINE_boolean('include_car', False, """Whether to include car in tracklet.""")
tf.app.flags.DEFINE_boolean('include_ped', False, """Whether to include pedestrian in tracklet.""")
tf.app.flags.DEFINE_boolean('enable_birdseye', False, """Whether to enable birdseye detection.""")
tf.app.flags.DEFINE_boolean('enable_camera', False, """Whether to enable camera detection.""")

def generate_predictions(pipeline, bag_file):
    generator = sensor.generate_sensormsgs(bag_file)
    for msg in generator:
        t_seconds = msg.header.stamp.to_sec()
        if msg.header.frame_id == 'camera':
            imagemsg = image.ImageMsg(msg)
            pipeline.detect_image(imagemsg.bgr, t_seconds)

            car, ped = pipeline.estimate_positions()
            yield car, ped

        elif msg.header.frame_id == 'velodyne':
            points = pc2.read_points(msg)
            points = np.array(list(points))
            pipeline.detect_lidar(points, t_seconds)

def predict_tracklet(bag_file, include_car, include_ped, enable_birdseye, enable_camera):
    pipeline = dp.DetectionPipeline(enable_birdseye, enable_camera)

    def make_pose(x, y, rz):
      # Estimate tz from histogram.
      return {'tx': x,
              'ty': y,
              'tz': -0.9,
              'rx': 0,
              'ry': 0,
              'rz': rz}

    prev_car_pose = make_pose(0, 0, 0)
    prev_ped_pose = make_pose(0, 0, 0)
    predicted_yaw = 0

    # l, w, h from histogram
    car_tracklet = generate_tracklet.Tracklet(object_type='Car', l=4.3, w=1.7, h=1.7, first_frame=0)
    ped_tracklet = generate_tracklet.Tracklet(object_type='Pedestrian', l=0.8, w=0.8, h=1.708, first_frame=0)

    generator = generate_predictions(pipeline, bag_file)

    for car, ped in generator:
        # car is tx, ty, tz, rz
        if car is not None:
            car_pose = make_pose(car[0], car[1], car[3])
            prev_car_pose = car_pose
        car_tracklet.poses.append(prev_car_pose)

        if ped is not None:
            ped_pose = make_pose(ped[0], ped[1], ped[3])
            prev_ped_pose = ped_pose
        ped_tracklet.poses.append(prev_ped_pose)

    tracklet_collection = generate_tracklet.TrackletCollection()
    if include_car:
        tracklet_collection.tracklets.append(car_tracklet)
    if include_ped:
        tracklet_collection.tracklets.append(ped_tracklet)

    tracklet_file = os.path.join(FLAGS.out_dir, get_filename(bag_file) + '.xml')

    tracklet_collection.write_xml(tracklet_file)

def get_filename(bag_file):
    base = os.path.basename(bag_file)
    split = os.path.splitext(base)
    return split[0]

def process_bag(bag_file):
    print('Generate tracklet')
    print('Include car: ', FLAGS.include_car)
    print('Include ped: ', FLAGS.include_ped)
    print('Enable birdseye: ', FLAGS.enable_birdseye)
    print('Enable camera: ', FLAGS.enable_camera)
    if not FLAGS.include_car and not FLAGS.include_ped:
        print('Must include one of the obstacle types.')
        exit()
    if not FLAGS.enable_birdseye and not FLAGS.enable_camera:
        print('Must enable one or more of the detectors.')
        exit()

    predict_tracklet(bag_file, FLAGS.include_car, FLAGS.include_ped, FLAGS.enable_birdseye, FLAGS.enable_camera)

def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    if FLAGS.bag_file:
        print('Processing single bag. {}'.format(FLAGS.bag_file))
        process_bag(FLAGS.bag_file)
    elif FLAGS.bag_dir:
        print('Processing bag folder. {}'.format(FLAGS.bag_dir))
        bags = bu.find_bags(FLAGS.bag_dir)
        for bag in bags:
            process_bag(bag)
    else:
        print('Neither bag_file nor bag_dir specified.')

if __name__ == '__main__':
    tf.app.run()
