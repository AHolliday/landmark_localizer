# Make a copy of this file called 'constants.py' in this location, and set all
# directory variables in constants.py according to your system.

import os
from pathlib import Path
class Dataset:
    pass


mtl_dir = Path('/home/a.holliday/montreal_dataset')

# sub-sets of the mtl dataset.
trottier1 = Dataset()
trottier1.dir = mtl_dir / 'nearTrottier1'
# trottier1.images = [os.path.join(trottier1.dir, f)
#                     for f in ['far.png', 'near.png']]

trottier2 = Dataset()
trottier2.dir = mtl_dir / 'nearTrottier2'
# trottier2.images = [os.path.join(trottier2.dir, f)
#                     for f in ['far.png', 'near.png']]

cartier = Dataset()
cartier.dir = mtl_dir / 'cartierMonument'

cathedral = Dataset()
cathedral.dir = mtl_dir / 'cathedral'

foodtruck = Dataset()
foodtruck.dir = mtl_dir / 'foodTruck'

icecream = Dataset()
icecream.dir = mtl_dir / 'iceCream'

mcconnell = Dataset()
mcconnell.dir = mtl_dir / 'mcconnell'

parkShack = Dataset()
parkShack.dir = mtl_dir / 'parkShack'

placeDesArts = Dataset()
placeDesArts.dir = mtl_dir / 'placeDesArtsStatues'

rowHomes1 = Dataset()
rowHomes1.dir = mtl_dir / 'rowHomes1'

rowHomes2 = Dataset()
rowHomes2.dir = mtl_dir / 'rowHomes2'

uqam = Dataset()
uqam.dir = mtl_dir / 'uqam'

parkingLot1 = Dataset()
parkingLot1.dir = mtl_dir / 'parkingLot1'

parkingLot2 = Dataset()
parkingLot2.dir = mtl_dir / 'parkingLot2'

stLaurent = Dataset()
stLaurent.dir = mtl_dir / 'stLaurent'

# simpleDatasets = [trottier1, trottier2, labDoor, posterTest]

# real-world sets
outdoorDatasets = [
    trottier1,
    trottier2,
    # cartier,
    cathedral,
    foodtruck,
    icecream,
    # parkShack,
    # placeDesArts,
    rowHomes1,
    rowHomes2,
    uqam,
    mcconnell,
    # parkingLot1,
    parkingLot2,
    stLaurent,
]

# initialize the outdoor datasets
def initOutdoorSet(outdoorSet):
    pngs = [f for f in os.listdir(outdoorSet.dir) if f.endswith('.png')]
    numPngs = [f for f in pngs if f.rpartition('.png')[0].isdigit()]
    # for png in pngs:
    #     try:
    #         # filter out any .pngs with any non-numeric characters
    #         int(png.rpartition('.png')[0])
    #         numPngs.append(png)
    #     except ValueError:
    #         continue
    numPngPaths = [os.path.join(outdoorSet.dir, f) for f in numPngs]
    outdoorSet.images = sorted(numPngPaths)

try:
    list(map(initOutdoorSet, outdoorDatasets))
except:
    # if the paths to the data are incorrect, proceed anyway.  We don't want
    # using any of the module to fail just because the data for one script
    # can't be found.
    pass


# coral dataset
coralDir = '/localdata/aholliday/thesis_data/coral_data_jan_2016'

defaultKittiDir = '/home/andrew/kitti/dataset'
defaultMaxKittiSpan = 10
earthRadiusM = 6371000

defaultColdDir = '/home/a.holliday/COLD/saarbrucken/seq2_cloudy1'

# keys for place recognition results
STEP_KEY = 'Step distance (m)'
DIST_KEY = 'Distance to match (m)'
NEAR_DIST_KEY = 'Distance to nearest (m)'
EST_DIST_KEY = 'Estimated distance (m)'
SYM_ERR_KEY = 'Pose error (m)'
SIM_KEY = 'Similarity score'
I_DIST_KEY = 'Index dist.'
NI_DIST_KEY = 'Normalized index dist.'
NEAR_I_DIST_KEY = 'Nearest index dist.'
NEW_PROB_KEY = 'New scene probability'
QROOM_KEY = 'Query room ID'
QROOM_TYPE_KEY = 'Query room type'
MROOM_KEY = 'Match room ID'
MROOM_TYPE_KEY = 'Match room type'


NET_LAYER_SIZES = {
 'alexnet': {'conv1': 14400,
             'conv2': 9408,
             'conv3': 3456,
             'conv4': 2304,
             'conv5': 2304,
             'pool1': 3136,
             'pool2': 1728,
             'pool5': 256},
 'densenet121': {'conv0': 65536,
                 'denseblock1': 65536,
                 'denseblock2': 32768,
                 'denseblock3': 16384,
                 'denseblock4': 4096,
                 'norm0': 65536,
                 'norm5': 4096,
                 'pool0': 16384,
                 'relu0': 65536,
                 'transition1': 8192,
                 'transition2': 4096,
                 'transition3': 2048},
 'densenet161': {'conv0': 98304,
                 'denseblock1': 98304,
                 'denseblock2': 49152,
                 'denseblock3': 33792,
                 'denseblock4': 8832,
                 'norm0': 98304,
                 'norm5': 8832,
                 'pool0': 24576,
                 'relu0': 98304,
                 'transition1': 12288,
                 'transition2': 6144,
                 'transition3': 4224},
 'densenet169': {'conv0': 65536,
                 'denseblock1': 65536,
                 'denseblock2': 32768,
                 'denseblock3': 20480,
                 'denseblock4': 6656,
                 'norm0': 65536,
                 'norm5': 6656,
                 'pool0': 16384,
                 'relu0': 65536,
                 'transition1': 8192,
                 'transition2': 4096,
                 'transition3': 2560},
 'densenet201': {'conv0': 65536,
                 'denseblock1': 65536,
                 'denseblock2': 32768,
                 'denseblock3': 28672,
                 'denseblock4': 7680,
                 'norm0': 65536,
                 'norm5': 7680,
                 'pool0': 16384,
                 'relu0': 65536,
                 'transition1': 8192,
                 'transition2': 4096,
                 'transition3': 3584},
 'resnet101': {'pool1': 16384,
               'pool5': 8192,
               'res2c': 65536,
               'res3d': 32768,
               'res4f': 16384,
               'res5c': 8192},
 'resnet152': {'pool1': 16384,
               'pool5': 8192,
               'res2c': 65536,
               'res3d': 32768,
               'res4f': 16384,
               'res5c': 8192},
 'resnet50': {'pool1': 16384,
              'pool5': 8192,
              'res2c': 65536,
              'res3d': 32768,
              'res4f': 16384,
              'res5c': 8192},
 'vgg11': {'pool1': 65536,
           'pool2': 32768,
           'pool3': 16384,
           'pool4': 8192,
           'pool5': 2048,
           'pre_pool1': 262144,
           'pre_pool2': 131072,
           'pre_pool3': 65536,
           'pre_pool4': 32768,
           'pre_pool5': 8192},
 'vgg13': {'pool1': 65536,
           'pool2': 32768,
           'pool3': 16384,
           'pool4': 8192,
           'pool5': 2048,
           'pre_pool1': 262144,
           'pre_pool2': 131072,
           'pre_pool3': 65536,
           'pre_pool4': 32768,
           'pre_pool5': 8192},
 'vgg16': {'pool1': 65536,
           'pool2': 32768,
           'pool3': 16384,
           'pool4': 8192,
           'pool5': 2048,
           'pre_pool1': 262144,
           'pre_pool2': 131072,
           'pre_pool3': 65536,
           'pre_pool4': 32768,
           'pre_pool5': 8192},
 'vgg19': {'pool1': 65536,
           'pool2': 32768,
           'pool3': 16384,
           'pool4': 8192,
           'pool5': 2048,
           'pre_pool1': 262144,
           'pre_pool2': 131072,
           'pre_pool3': 65536,
           'pre_pool4': 32768,
           'pre_pool5': 8192}
 }

LAYER_ORDERS = {
    'alexnet': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4',
                'conv5', 'pool5'],
    'vgg': ['pre_pool1', 'pool1', 'pre_pool2', 'pool2', 'pre_pool3',
            'pool3', 'pre_pool4', 'pool4', 'pre_pool5', 'pool5'],
    'resnet': ['pool1', 'res2c', 'res3d', 'res4f', 'res5c', 'pool5'],
    'densenet': ['denseblock1', 'transition1', 'denseblock2',
                 'transition2', 'denseblock3', 'transition3',
                 'denseblock4']
}
