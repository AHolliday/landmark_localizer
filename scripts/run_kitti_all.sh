CFG=config/kitti/eb_dense169_d2sub.yaml
KITTI_DIR=/usr/local/data/ahollid/kitti/

# including the training set
# python scripts/sequence_experiment.py --cfg $CFG \
# --list ../kitti/all00_sub5.txt \
# --list ../kitti/all01_sub5.txt \
# --list ../kitti/all04_sub5.txt \
# --list ../kitti/all09_sub5.txt \
# --list ../kitti/all02_sub5.txt \
# --list ../kitti/all03_sub5.txt \
# --list ../kitti/all06_sub5.txt \
# --list ../kitti/all10_sub5.txt \
# --list ../kitti/all05_sub5.txt \
# --list ../kitti/all07_sub5.txt \
# --list ../kitti/all08_sub5.txt &

# not including the training set
python scripts/sequence_experiment.py --cfg $CFG \
--list $KITTI_DIR/all05_sub5.txt \
--list $KITTI_DIR/all08_sub5.txt &

python scripts/sequence_experiment.py --cfg $CFG \
--list $KITTI_DIR/all00_sub5.txt \
--list $KITTI_DIR/all03_sub5.txt \
--list $KITTI_DIR/all10_sub5.txt &

python scripts/sequence_experiment.py --cfg $CFG \
--list $KITTI_DIR/all02_sub5.txt \
--list $KITTI_DIR/all04_sub5.txt \
--list $KITTI_DIR/all09_sub5.txt &
