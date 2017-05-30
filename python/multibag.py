import fnmatch
import os
import sys
import traindata

sys.path.append('/home/eljefec/repo/didi-competition/tracklets/python')
import bag_utils

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

def find_bag_tracklets(directory, pattern, tracklet_dir):
    bag_tracklets = []

    bags = find_bags(directory, pattern)
    assert(len(bags) > 0)

    for bag in bags:
        tracklet_path = find_tracklet(bag, tracklet_dir)
        if tracklet_path is not None:
            bag_tracklets.append(BagTracklet(bag, tracklet_path))

    return bag_tracklets

def count_image_messages(bag_dir):
    count = 0
    bagsets = bag_utils.find_bagsets(bag_dir)
    for bs in bagsets:
        count += bs.get_message_count(['/image_raw'])
    return count

class MultiBagStream:
    def __init__(self, bag_dir, tracklet_dir):
        self.bag_tracklets = find_bag_tracklets(bag_dir, '*.bag', tracklet_dir)
        self.traindata = traindata.TrainDataStream()
        self.frame_count = count_image_messages(bag_dir)
        self.bag_index = 0

    def next(self):
        if self.traindata.empty():
            self.bag_index = (self.bag_index + 1) % len(self.bag_tracklets)
            bag_tracklet = self.bag_tracklets[self.bag_index]
            self.traindata.start_read(bag_tracklet.bag, bag_tracklet.tracklet)

        return self.traindata.next()

if __name__ == '__main__':
    bags = find_bags('/data/Didi-Release-2/Data/', '*.bag')
    print(bags)
    for b in bags:
        print('basename: ', os.path.basename(b))
        print('dirname: ', os.path.dirname(b))
        print('split: ', os.path.split(b))
        print('splitext: ', os.path.splitext(b))

    bag_tracklets = find_bag_tracklets('/data/Didi-Release-2/Data/', '*.bag', '/data/output/tracklet')
    for bt in bag_tracklets:
        print(bt)

    multibag = MultiBagStream('/data/Didi-Release-2/Data/', '/data/output/tracklet')
    print(multibag.frame_count)
    for i in range(5):
        msg = multibag.next()
        print('got msg: ', msg)
