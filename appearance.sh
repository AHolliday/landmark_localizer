WEBCAMDIR=../WebcamRelease

# python scripts/appearance_experiment.py \
# $WEBCAMDIR/Chamonix/test/image_color/ \
# --cfg config/kitti/resnet_redux.yaml &
#
# python scripts/appearance_experiment.py \
# $WEBCAMDIR/Courbevoie/test/image_color/ \
# --cfg config/kitti/resnet_redux.yaml &

# python scripts/appearance_experiment.py \
# $WEBCAMDIR/Frankfurt/test/image_color/ \
# --cfg config/kitti/resnet_redux.yaml &
#
# python scripts/appearance_experiment.py \
# $WEBCAMDIR/Mexico/test/image_color/ \
# --cfg config/kitti/resnet_redux.yaml &
#
python scripts/appearance_experiment.py \
$WEBCAMDIR/StLouis/test/image_color/ \
--cfg config/kitti/resnet_redux.yaml &
