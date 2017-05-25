import numpy
import pickle
with open('pointcloud.p', 'rb') as f:
    lidar = pickle.load(f, encoding='latin1')
    print(lidar.shape)
    print(lidar[1:2])
