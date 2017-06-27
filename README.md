# README #

### What is this repository for? ###

This contains my work (Team Eva) on the Udacity-Didi competition (https://www.udacity.com/didi-challenge)

### Scripts

#### lidarbag.py
- Convert ROS bags containing `velodyne_packets` messages to separate bags containing `velodyne_points` messages
#### lidar.py
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
#### framestream.py
- `def generate_trainmsgs(bag, tracklet)`
- Synchronize image and lidar messages
- Combine latest image and lidar into single `TrainMsg`
- Returns `class TrainMsg`
    - pose
    - image
    - lidar
#### sensor.py
- `def generate_sensormsgs(bag_file)`
- Process images and lidar from a ROS bag
- Return image and lidar messages in sequence
