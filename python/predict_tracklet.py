import generate_tracklet
import generator
import keras
import multibag
import pandas as pd
import pickle
import picklebag

# From https://github.com/udacity/didi-competition/blob/master/tracklets/python/bag_utils.py
def load_metadata(metadata_path):
    metadata_df = pd.read_csv(metadata_path, header=0, index_col=None, quotechar="'")
    return metadata_df.to_dict(orient='records')

# From https://github.com/udacity/didi-competition/blob/master/tracklets/python/bag_to_kitti.py
def extract_metadata(md, obs_name):
    md = next(x for x in md if x['obstacle_name'] == obs_name)
    return md

def numpy_preds_to_dicts(predictions):
    poses = []
    count = predictions.shape[0]
    for i in range(count):
        # See framestream.py
        # 0-2: l,w,h
        # 3-5: tx, ty, tz
        # 6-8: rots
        np_pred = predictions[i]
        pose = {'tx': np_pred[3],
                'ty': np_pred[4],
                'tz': np_pred[5],
                'rx': 0,
                'ry': 0,
                'rz': 0}
        poses.append(pose)
    return poses

def predict_tracklet(model_file, bag_file, metadata_path, output_file, pickle_file = None):
    md = extract_metadata(load_metadata(metadata_path), 'obs1')
    tracklet = generate_tracklet.Tracklet(object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

    model = keras.models.load_model(model_file)

    # datastream = picklebag.PickleAdapter()
    # datastream.start_read(bag_file, tracklet_file = None)
    datastream = multibag.MultiBagStream([multibag.BagTracklet(bag_file, None)],
                                         use_pickle_adapter = False)
    input_generator = generator.TrainDataGenerator(datastream, include_ground_truth = False)

    batch_size = 64
    print('input_generator.get_count()', input_generator.get_count())
    steps = input_generator.get_count() / batch_size
    if input_generator.get_count() % batch_size > 0:
        steps += 1
    print('steps', steps)
    predictions = model.predict_generator(input_generator.generate(batch_size), steps)

    print('predictions.shape:', predictions.shape)

    # Remove extra predictions due to using batch-based input_generator.
    predictions = predictions[:input_generator.get_count()]

    print('predictions.shape:', predictions.shape)

    tracklet.poses = numpy_preds_to_dicts(predictions)
    tracklet_collection = generate_tracklet.TrackletCollection()
    tracklet_collection.tracklets.append(tracklet)
    tracklet_collection.write_xml(output_file)

    if pickle_file:
        result_map = dict()
        result_map['model_file'] = model_file
        result_map['bag_file'] = bag_file
        result_map['predictions'] = predictions

        with open(pickle_file, 'wb') as f:
            pickle.dump(result_map, f)

if __name__ == '__main__':
    model_file = '/home/eljefec/repo/eva-didi/python/checkpoints/model_2017-06-01_11h24m33e04-vl100.04.h5'
    predict_tracklet(model_file,
                     '/data/Didi-Release-2/Data/Round 1 Test/19_f2.bag',
                     '/data/Didi-Release-2/Data/Round 1 Test/metadata.csv',
                     '/data/output/test/19_f2.conv3d.predicted_tracklet.xml',
                     '/data/output/test/predictions.p')
