import os
import pickle
import traindata

class FramePickle:
    def __init__(self, pickle_filename):
        self.pickle_filename = pickle_filename

    def generate(self):
        with open(self.pickle_filename, 'rb') as f:
            frames = pickle.load(f)
        for frame in frames:
            yield frame

    def dump(self, frames):
        print('DEBUG: Dumping frames to pickle. ', self.pickle_filename)
        with open(self.pickle_filename, 'wb') as f:
            pickle.dump(frames, f)

def get_pickle_filename(bag_file, pickle_id):
    return bag_file + '.' + str(pickle_id) + '.p'

HEADER_ID = 'header'
FRAME_COUNT = 'frame_count'
FRAME_FILENAMES = 'frame_filenames'

class PickleAdapter:
    def __init__(self, frames_per_pickle = 50):
        self.frames_per_pickle = frames_per_pickle
        self.frame_count = None
        self.generator = None
        self.next_frame = None

    def start_read(self, bag_file, tracklet_file):
        header_file = get_pickle_filename(bag_file, HEADER_ID)
        if os.path.exists(header_file):
            print('DEBUG: header found')
        else:
            print('DEBUG: header not found. Dicing pickles.')
            split_into_pickles(bag_file, tracklet_file, header_file, self.frames_per_pickle)

        self.generator = self.generate(header_file)
        try:
            self.next_frame = next(self.generator)
        except StopIteration:
            self.next_frame = None
            self.generator = None

    def empty(self):
        return self.next_frame is None

    def next(self):
        if self.next_frame is None:
            return None
        else:
            current_frame = self.next_frame
            try:
                self.next_frame = next(self.generator)
            except StopIteration:
                self.next_frame = None
                self.generator = None
            return current_frame

    def generate(self, header_file):
        with open(header_file, 'rb') as f:
            header = pickle.load(f)
            self.frame_count = header[FRAME_COUNT]
            frame_filenames = header[FRAME_FILENAMES]
        frame_pickles = []
        for frame_filename in frame_filenames:
            assert(os.path.exists(frame_filename))
            frame_pickles.append(FramePickle(frame_filename))
        for frame_pickle in frame_pickles:
            generator = frame_pickle.generate()

            empty = False
            while not empty:
                try:
                    yield next(generator)
                except StopIteration:
                    empty = True

def split_into_pickles(bag_file, tracklet_file, header_file, frames_per_pickle):
    frame_count = 0
    frames = []
    pickles = []
    datastream = traindata.TrainDataStream()
    datastream.start_read(bag_file, tracklet_file)
    while not datastream.empty():
        train_data_frame = datastream.next()
        frames.append(train_data_frame)
        frame_count += 1
        if (frame_count % frames_per_pickle) == 0:
            # Dump pickle.
            frame_pickle = FramePickle(get_pickle_filename(bag_file, len(pickles)))
            frame_pickle.dump(frames)
            frames = []
            pickles.append(frame_pickle)
    # Dump rest of frames to pickle.
    frame_pickle = FramePickle(get_pickle_filename(bag_file, len(pickles)))
    frame_pickle.dump(frames)
    frames = []
    pickles.append(frame_pickle)
    # Pickle header
    frame_filenames = []
    for frame_pickle in pickles:
        frame_filenames.append(frame_pickle.pickle_filename)
    header = dict()
    header[FRAME_COUNT] = frame_count
    header[FRAME_FILENAMES] = frame_filenames
    with open(header_file, 'wb') as f:
        pickle.dump(header, f)

if __name__ == '__main__':
    count = 0
    pickle_adapter = PickleAdapter()
    for i in range(2):
        pickle_adapter.start_read('/data/Didi-Release-2/Data/1/2.bag', '/data/output/test/2/tracklet_labels.xml')
        while not pickle_adapter.empty():
            td = pickle_adapter.next()
            count += 1
            print(count)
    print('count: {0}'.format(count))
