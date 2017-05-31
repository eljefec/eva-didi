from __future__ import division

import fnmatch
import os
import random
import rosbag
import sys
import traindata

def find_bags(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
    return sorted(matched_files)

def find_tracklet(bag_file, tracklet_dir):
    for height in range(3):
        base = os.path.splitext(bag_file)[0]
        subdir_stack = []
        for i in range(height):
            split = os.path.split(base)
            base = split[0]
            subdir_stack.append(split[1])
        tracklet_path = tracklet_dir
        while subdir_stack:
            tracklet_path = os.path.join(tracklet_path, subdir_stack.pop())
        tracklet_path = os.path.join(tracklet_path, 'tracklet_labels.xml')
        if os.path.exists(tracklet_path):
            return tracklet_path

    return None

class BagTracklet:
    def __init__(self, bag, tracklet):
        self.bag = bag
        self.tracklet = tracklet

    def __repr__(self):
        return 'bag: {0}, tracklet: {1}'.format(self.bag, self.tracklet)

def find_bag_tracklets(directory, tracklet_dir):
    bag_tracklets = []

    bags = find_bags(directory, '*.bag')
    assert(len(bags) > 0)

    for bag in bags:
        tracklet_path = find_tracklet(bag, tracklet_dir)
        if tracklet_path is not None:
            bag_tracklets.append(BagTracklet(bag, tracklet_path))

    return bag_tracklets

def count_image_messages(bag_tracklets):
    total = 0
    for bt in bag_tracklets:
        bag = rosbag.Bag(bt.bag, 'r')
        total += bag.get_message_count(topic_filters=['/image_raw'])
        bag.close()
    return total

def count_image_messages_per_bag(bag_tracklets):
    counts = []
    for bt in bag_tracklets:
        bag = rosbag.Bag(bt.bag, 'r')
        counts.append(bag.get_message_count(topic_filters=['/image_raw']))
        bag.close()
    return counts

class TrainValidationSplit:
    def __init__(self, train_bags, train_count, validation_bags, validation_count):
        self.train_bags = train_bags
        self.train_count = train_count
        self.validation_bags = validation_bags
        self.validation_count = validation_count

    def __repr__(self):
        return 'train: {} count: {}\nvalidation: {} count: {}'.format(self.train_bags, self.train_count, self.validation_bags, self.validation_count)

def train_validation_split(bag_tracklets, validation_split):
    assert(validation_split >= 0 and validation_split <= 1)

    counts = count_image_messages_per_bag(bag_tracklets)
    total = sum(counts)

    partial_sum = 0
    for i in range(len(bag_tracklets)):
        partial_sum += counts[i]
        validation_percent = partial_sum / total
        if (validation_percent >= validation_split):
            validation_bags = bag_tracklets[0:i+1]
            train_bags = bag_tracklets[i+1:]
            break

    return TrainValidationSplit(train_bags, (total - partial_sum), validation_bags, partial_sum)

def shuffle(bag_tracklets, seed):
    random.seed(seed)
    random.shuffle(bag_tracklets)

class MultiBagStream:
    def __init__(self, bag_tracklets):
        self.bag_tracklets = bag_tracklets
        self.traindata = traindata.TrainDataStream()
        self.frame_count = count_image_messages(bag_tracklets)
        self.bag_index = 0

    def count(self):
        return self.frame_count

    def next(self):
        if self.traindata.empty():
            self.bag_index = (self.bag_index + 1) % len(self.bag_tracklets)
            bag_tracklet = self.bag_tracklets[self.bag_index]
            print('Opening next bag: ', bag_tracklet.bag)
            self.traindata.start_read(bag_tracklet.bag, bag_tracklet.tracklet)

        return self.traindata.next()

if __name__ == '__main__':
    import copy

    bag_tracklets = find_bag_tracklets('/data/Didi-Release-2/Data/', '/data/output/tracklet')
    for bt in bag_tracklets:
        print(bt)

    for seed in range(10):
        copied = copy.copy(bag_tracklets)
        shuffle(copied, seed)
        split = train_validation_split(copied, 0.15)
        print('seed: ', seed)
        print('split: ', split)

    shuffle(bag_tracklets, 7)
    split = train_validation_split(bag_tracklets, 0.15)
    print('split2: ', split)

    multibag = MultiBagStream(split.train_bags)
    print('train:', multibag.count())

    multibag = MultiBagStream(split.validation_bags)
    print('validation:', multibag.count())

    #for i in range(5):
    #    msg = multibag.next()
    #    print('got msg: ', msg)
