import cv2
import numpy as np
import os
import yaml

import camera_converter as cc
import multibag as mb
import numpystream as ns
import squeezedet
import util.traingen

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

if __name__ == '__main__':
    bag_tracklets = mb.find_bag_tracklets('/data/bags/', '/data/tracklets')
    write_training_data(bag_tracklets, '/home/eljefec/repo/squeezeDet/data/KITTI/camera_train')
    exit()

    import os
    path = '/data/dev/camera'
    if not os.path.exists(path):
        os.makedirs('/data/dev/camera')
    try_undistort(None)
