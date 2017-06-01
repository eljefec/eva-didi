import generate_tracklet as gt
import my_bag_utils as bu
import numpy as np
import predict_tracklet as pt

# Tool for manually creating a tracklet with dummy data.
def manual_tracklet(bag_file, metadata_path, output_file):
    count = bu.count_image_msgs(bag_file)
    tx = 0
    ty = 0
    tz = 0
    pred = np.array([[0, 0, 0, tx, ty, tz, 0, 0, 0]])
    predictions = np.repeat(pred, count, axis=0)

    print(predictions.shape)

    md = pt.extract_metadata(pt.load_metadata(metadata_path), 'obs1')
    tracklet = gt.Tracklet(object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

    tracklet.poses = pt.numpy_preds_to_dicts(predictions)
    tracklet_collection = gt.TrackletCollection()
    tracklet_collection.tracklets.append(tracklet)
    tracklet_collection.write_xml(output_file)

if __name__ == '__main__':
    manual_tracklet('/data/Didi-Release-2/Data/1/2.bag',
                    '/data/Didi-Release-2/Data/1/metadata.csv',
                    '/data/output/test/2.predicted_tracklet.xml')
