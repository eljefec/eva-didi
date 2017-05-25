import numpy
import pickle
with open('pointcloud.p', 'rb') as f:
    lidar = pickle.load(f, encoding='latin1')
    print(lidar.shape)
    print(lidar[1:2])
with open('header.p', 'rb') as f:
    header = pickle.load(f)
    print(header)
