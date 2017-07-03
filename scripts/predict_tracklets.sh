#!/bin/sh

die () {
    echo >&2 "$@"
    echo "Usage: sh ./scripts/predict_tracklets.sh RUN_NAME"
    exit 1
}

[ "$#" -eq 1 ] || die "1 argument required, $# provided"

RUN_NAME=$1

DIDI_DIR=/home/eljefec/repo/didi-competition/tracklets/python
OUT_DIR=/data/out/$RUN_NAME

# Generate tracklets for test set.
python python/predict_tracklet.py --bag_dir /data/bags/didi-round2/release/car/testing --do tracklet --include_car --out_dir $OUT_DIR/test
python python/predict_tracklet.py --bag_file /data/bags/didi-round2/release/pedestrian/ped_test.bag --do tracklet --include_ped --out_dir $OUT_DIR/test

# Generate tracklets for training set.
python python/predict_tracklet.py --bag_file /data/bags/didi-round2/release/car/training/suburu_leading_front_left/suburu11.bag --do tracklet --include_car --out_dir $OUT_DIR
mkdir $OUT_DIR/suburu11
python $DIDI_DIR/evaluate_tracklets.py $OUT_DIR/suburu11.xml /data/tracklets/didi-round2_release_car_training_suburu_leading_front_left-suburu11/tracklet_labels.xml -o $OUT_DIR/suburu11

# Generate tracklets for pedestrian training set.
python python/predict_tracklet.py --bag_file /data/bags/didi-round2/release/pedestrian/ped_train.bag --do tracklet --include_ped --out_dir $OUT_DIR
mkdir $OUT_DIR/ped_train
python $DIDI_DIR/evaluate_tracklets.py $OUT_DIR/ped_train.xml /data/tracklets/didi-round2_release_pedestrian-ped_train/tracklet_labels.xml -o $OUT_DIR/ped_train

# Generate tracklets for long bag.
python python/predict_tracklet.py --bag_file /data/bags/didi-round2/release/car/training/bmw_following_long/bmw02.bag --do tracklet --include_car --out_dir $OUT_DIR
mkdir $OUT_DIR/bmw02
python $DIDI_DIR/evaluate_tracklets.py $OUT_DIR/bmw02.xml /data/tracklets/didi-round2_release_car_training_bmw_following_long-bmw02/tracklet_labels.xml -o $OUT_DIR/bmw02
