from __future__ import division

import my_bag_utils as bu
import os
import random
import rosbag
import sys
import traindata

def find_tracklet(bag_file, tracklet_dir):
    for height in range(1, 10):
        base = os.path.splitext(bag_file)[0]
        subdir_stack = []
        for i in range(height):
            split = os.path.split(base)
            base = split[0]
            subdir_stack.append(split[1])
        tracklet_path = tracklet_dir

        # Reassemble the tracklet path created by Udacity's bag_to_kitti.py
        subdir = ''
        while len(subdir_stack) > 1:
            if subdir:
                subdir += '_'
            subdir += subdir_stack.pop()

        # Bag name
        subdir += '-' + subdir_stack.pop()

        tracklet_path = os.path.join(tracklet_path, subdir)
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

    bags = bu.find_bags(directory)
    assert(len(bags) > 0)

    for bag in bags:
        tracklet_path = find_tracklet(bag, tracklet_dir)
        if tracklet_path is not None:
            bag_tracklets.append(BagTracklet(bag, tracklet_path))

    return bag_tracklets

def count_image_messages(bag_tracklets):
    total = 0
    for bt in bag_tracklets:
        total += bu.count_image_msgs(bt.bag)
    return total

def count_image_messages_per_bag(bag_tracklets):
    counts = []
    for bt in bag_tracklets:
        counts.append(bu.count_image_msgs(bt.bag))
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
    def __init__(self, bag_tracklets, fn_create_generator = traindata.generate_traindata):
        self.bag_tracklets = bag_tracklets
        self.frame_count = count_image_messages(bag_tracklets)
        self.fn_create_generator = fn_create_generator

        self.bag_index = 0

    def count(self):
        return self.frame_count

    def generate(self, infinite):
        bag_index = 0
        bag_tracklet = self.bag_tracklets[bag_index]
        generator = self.fn_create_generator(bag_tracklet.bag, bag_tracklet.tracklet)
        while True:
            try:
                yield next(generator)
            except StopIteration:
                bag_index += 1
                if not infinite:
                    if bag_index >= len(self.bag_tracklets):
                        raise StopIteration
                bag_index = bag_index % len(self.bag_tracklets)
                bag_tracklet = self.bag_tracklets[bag_index]
                print('Opening next bag: ', bag_tracklet.bag)
                generator = self.fn_create_generator(bag_tracklet.bag, bag_tracklet.tracklet)

if __name__ == '__main__':
    import copy

    bag_tracklets = find_bag_tracklets('/data/bags/', '/data/tracklets')
    for bt in bag_tracklets:
        print(bt)

    stream = MultiBagStream(bag_tracklets)
    print(stream.count())
    count = 0
    for datum in stream.generate():
        count += 1
        if count % 1000 == 0:
            print(count)

    exit()

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
