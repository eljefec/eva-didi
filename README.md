# README #

### What is this repository for? ###

This contains my work (Jeffrey Liu of Team Eva) on the Udacity-Didi competition (https://www.udacity.com/didi-challenge)

### Example Output

[//]: # (Image References)

[car_detection]: ./README/car_detection.png "Car Detection"
[ped_detection]: ./README/ped_detection.png "Pedestrian Detection"

#### Car Detection

This shows a car detection in a bird's eye view of lidar data. The blue rectangle indicates a detected car, and the label shows a confidence of 0.83.

![alt_text][car_detection]

#### Pedestrian Detection

This shows a pedestrian detection in a bird's eye view of lidar data. The pink square indicates a detected pedestrian, and the label shows a confidence of 0.57.

![alt_text][ped_detection]

#### Video

[ford06.mp4](https://github.com/eljefec/eva-didi/blob/master/README/ford06.mp4) is a video of obstacle detections over a bird's eye view of lidar data. Car and pedestrian detections are marked by bounding boxes with a class and confidence label.

### Dependencies

- Fork of squeezeDet
    - Add to PYTHONPATH
    - https://github.com/eljefec/squeezeDet
    - Original: https://github.com/BichenWuUCB/squeezeDet
- Fork of didi-competition
    - Add to PYTHONPATH
    - https://github.com/eljefec/didi-competition
    - Original: https://github.com/udacity/didi-competition
- Install ROS
- Install conda environment with `environment-gpu.yml`

### File Overview

#### ros_node.py
- ROS node that subscribes to messages (e.g., lidar) and runs them through `class DetectionPipeline`
- Dependencies:
    - velodyne driver must be running
    - Command: `roslaunch velodyne_pointcloud 32e_points.launch`
    - ROS master node must be running
    - Command: `roscore`
#### run_squeezedet.py
- Old script for predicting tracklets and making videos of obstacle detections
#### predict_tracklet.py
- Latest script for predicting tracklets
#### detection_pipeline.py
- `class DetectionPipeline`
- Combines different obstacle detectors into a single pipeline
#### birdseye_detector.py
- `class BirdsEyeDetector`
- Detects cars and pedestrians in a bird's eye view of lidar
- This performs well
#### squeezedet.py
- `class SqueezeDetector`
- Performs 2D bounding box detection on trained squeezeDet models
#### generate_kitti.py
- `def generate_kitti(...)`
- Generates training data for `class BirdsEyeDetector` in format expected by squeezeDet training scripts
#### rotation_predictor.py
- `class RotationPredictor`
- Predicts rotation of obstacle from bird's eye view of lidar
- This performs well
#### panorama_detector.py
- `class PanoramaDetector`
- Detects cars and pedestrians in a panorama view of lidar
- This is not finished
#### radar_detector.py
- `class RadarDetector`
- This is not finished
#### camera_detector.py
- `class CameraDetector`
- Detects cars and pedestrians from camera images
- This does not perform well
- `class ImageBoxToPosePredictor`
- Predicts pose of obstacle based on 2D bounding box
- This is a trained neural network
#### kalman_filter.py
- `class KalmanFilter`
- Unscented kalman filter
- This reduced accuracy of tracklets, but there may be some flaw in its implementation
#### lidarbag.py
- Convert ROS bags containing `velodyne_packets` messages to separate bags containing `velodyne_points` messages
#### lidar.py
- `def lidar_to_panorama(lidar)`
- `def lidar_to_slices(lidar)`
- `def lidar_to_birdseye(lidar)`
- `class PointCloudProcessor`
    - Clients can register to be called back with PointCloud2 messages
    - ROS node
    - Publish `velodyne_packets` messages for conversion by `velodyne_pointcloud` node
    - Subscribe to `velodyne_points` messages
#### multibag.py
- Read multiple bags
- Returns `class TrainData`
```
bag_tracklets = find_bag_tracklets(bag_dir, tracklet_dir)
stream = MultiBagStream(bag_tracklets)
for datum in stream.generate():
    count += 1
```
#### traindata.py
- `def generate_traindata(bag, tracklet)`
- Prepare training data
- Returns `class TrainData`
    - pose
    - image (BGR)
    - lidar (panorama)
    - lidar height slices
#### numpystream.py
- `def generate_numpystream(bag, tracklet)`
- Generate stream of numpy messages (converted from ROS messages)
- Returns `class NumpyData`
    - pose
    - image (BGR)
    - lidar (numpy)
#### framestream.py
- `def generate_trainmsgs(bag, tracklet)`
- Synchronize image and lidar ROS messages
- Combine latest image and lidar ROS messages into single `TrainMsg`
- Returns `class TrainMsg`
    - pose
    - image (ROS msg)
    - lidar (ROS msg)
#### sensor.py
- `def generate_sensormsgs(bag_file)`
- Process images and lidar from a ROS bag
- Return image and lidar messages in sequence
#### predict_tracklet.py
- `def predict_tracklet(bag_file)`
- Read lidar messages from bag and detect obstacles as bounding boxes
#### track.py
- `class Tracker`
- Track bounding boxes over time
