import pickle
import tempfile
import subprocess
import shlex
import os
from pathlib import Path
import numpy as np
import scipy.io
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import copy
import torch
from torchvision import transforms
# from pycocotools import mask as masklib
import h5py

from . import myCvTools as mycv
from . import geometryUtils as geo
from .pytorch_models import get_resnet_feat_extractor, \
    get_vgg_feat_extractor, get_densenet_feat_extractor, \
    get_alexnet_feat_extractor


# directories
LIB_STD_CPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
LOCAL_DIR = "/localdata/aholliday"

# RESNET_50_KEY = 'resnet50'
ALEX_KEY = 'alexnet'
VGG_KEYS = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
RESNET_KEYS = ['resnet50', 'resnet101', 'resnet152']
DENSE_KEYS = ['densenet121', 'densenet161', 'densenet169', 'densenet201']

# Other constant values
ABSTRACT_METHOD_CALLED_ERROR = AttributeError("The subclass on which this \
method was called must implement it, but does not!")


class Scene:
    @classmethod
    def fromHdf5(cls, hdf5Db, name):
        grp = hdf5Db[name]
        img = None
        if 'image' in grp:
            img = grp['image'][...]
        boxes = grp['boxes'][...]
        descriptors = grp['descriptors'][...]
        sub_feats = []
        if 'subFeatures' in grp:
            sub_grp = grp['subFeatures']
            sub_descs = sub_grp['descriptors'][...]
            kps = [cv2.KeyPoint(*rr[:-1], _octave=int(rr[-1]))
                   for rr in sub_grp['keypoints']]
            sub_feats = list(zip(kps, sub_descs))
        return cls(boxes, descriptors, img, subFeatures=sub_feats)


    def __init__(self, boxes, descriptors, img=None, masks=None,
                 subFeatures=None):
        self._img = img
        if type(boxes) is not np.ndarray:
            self._boxes = np.array(boxes)
        else:
            self._boxes = boxes
        if type(descriptors) is not np.ndarray:
            self._descriptors = np.array(descriptors)
        else:
            self._descriptors = descriptors
        self._masks = masks
        self._allSubFeatures = subFeatures

    def __len__(self):
        return len(self.boxes)

    def toHdf5(self, hdf5Db, name):
        # TODO we don't deal with masks for now
        grp = hdf5Db.create_group(name)
        if self.image is not None:
            grp.create_dataset('image', data=self.image)
        grp.create_dataset('boxes', data=self.boxes)
        grp.create_dataset('descriptors', data=self.descriptors)
        if self.hasSubFeatures:
            sub_grp = grp.create_group('subFeatures')
            kps, sub_descs = list(zip(*self.allSubFeatures))
            sub_grp.create_dataset('descriptors', data=np.array(sub_descs))
            pts = [(kp.pt) + (kp.size, kp.angle, kp.response, kp.octave)
                   for kp in kps]
            sub_grp.create_dataset('keypoints', data=np.array(pts))

    @property
    def objects(self):
        objects = []
        if self.hasSubFeatures:
            sub_pts = np.array([kp.pt for kp, _ in self._allSubFeatures])

        for bb, dd, mm in zip(self.boxes, self.descriptors, self.masks):
            object = WorldObject(bb, dd, mm,)
            if self.hasSubFeatures:
                inside_horiz = np.logical_and(sub_pts[:, 0] > bb[0],
                                              sub_pts[:, 0] < bb[2])
                inside_vert = np.logical_and(sub_pts[:, 1] > bb[1],
                                             sub_pts[:, 1] < bb[3])
                inside_mask = np.logical_and(inside_horiz, inside_vert)
                sub_idxs = np.where(inside_mask)[0]
                object.setSubFeatures(sub_idxs, self._allSubFeatures)
            objects.append(object)
        return objects

    @property
    def hasSubFeatures(self):
        return self._allSubFeatures is not None and \
            len(self._allSubFeatures) > 0

    @property
    def allSubFeatures(self):
        return self._allSubFeatures

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, img):
        self._img = img

    @property
    def boxes(self):
        return self._boxes

    @property
    def descriptors(self):
        return self._descriptors

    @property
    def hasMasks(self):
        return self._masks is not None and \
            any([mm is not None for mm in self._masks])

    @property
    def masks(self):
        if self._masks is None:
            return [None] * len(self)
        else:
            return self._masks


class WorldObject:
    def __init__(self, box, descriptor=None, mask=None, subFeatIdxs=[],
                 allFeatures=[]):
        self.box = box
        if type(mask) is np.ndarray:
            raise NotImplementedError('need pycocotools for this')
            # int_mask = mask.astype(np.uint8)
            # self._mask_hash = masklib.encode(np.asfortranarray(int_mask))
        elif mask is None or type(mask) is dict:
            self._mask_hash = mask
        else:
            raise ValueError('Inappropriate mask type ' + str(type(mask)) +
                             'received!')
        self.descriptor = descriptor
        self.setSubFeatures(subFeatIdxs, allFeatures)

    def setSubFeatures(self, subFeatIdxs, allFeatures):
        self.subFeatIdxs = subFeatIdxs
        self._global_sub_feats = allFeatures

    def getSubFeatures(self):
        return [self._global_sub_feats[ii] for ii in self.subFeatIdxs]

    def getSubFeatIdxs(self):
        return self.subFeatIdxs

    def containsPoint(self, x_coord, y_coord):
        return self.box[0] < x_coord and x_coord < self.box[2] and \
            self.box[1] < y_coord and y_coord < self.box[3]

    @property
    def allGlobalFeatures(self):
        return self._global_sub_feats

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def aspectRatio(self):
        return self.width / self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return geo.getCenterFromBox(self.box)
        # # could also try returning the centroids of the internal points, but
        # # there is danger of recursion here.
        # if len(self.subFeatures) > 0:
        #     return np.mean(self.points, axis=0)
        # else:
        #     return geo.getCenterFromBox(self.box)

    @property
    def points(self):
        sub_features = self.getSubFeatures()
        if len(sub_features) == 0:
            return [self.center]
        else:
            pts, _ = list(zip(*sub_features))
            return pts

    @property
    def mask(self):
        if self._mask_hash is None:
            return None
        # else:
        #     return masklib.decode(self._mask_hash)

    @property
    def encodedMask(self):
        return self._mask_hash


# Object extraction classes


class EdgeBoxExtractor:
    def __init__(self, model_path, **other_eb_args):
        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection(
            model_path)
        self.eb_engine = cv2.ximgproc.createEdgeBoxes(**other_eb_args)

    def get_boxes(self, image):
        # WARNING we don't use this anywhere right now.
        f32_image = np.float32(image) / 255.0
        edges = self.edge_detector.detectEdges(f32_image)
        orientation_map = self.edge_detector.computeOrientation(edges)
        edges = self.edge_detector.edgesNms(edges, orientation_map)
        boxes = self.eb_engine.getBoundingBoxes(edges, orientation_map)
        if len(boxes) == 0:
            # no boxes were detected!
            return np.zeros((0, 4))
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes


class SSExtractor:
    def __init__(self, scale, sigma, max_ratio=None, max_boxes=None,
                 min_area=1):
        # default min_area is 1 so that boxes of area 0 will be ruled out
        self.ss_scale = scale
        self.ss_sigma = sigma
        self.max_ratio = max_ratio
        self.max_boxes = max_boxes
        self.min_area = min_area
        self.box_engine = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def get_boxes(self, image):
        self.box_engine.setBaseImage(image)
        self.box_engine.switchToSingleStrategy(k=self.ss_scale,
                                               sigma=self.ss_sigma)
        boxes = self.box_engine.process()
        # print('started with', len(boxes), 'boxes')
        candidates = []
        # print('with ratio', self.max_ratio, 'start with', len(boxes))
        for box in boxes:
            _, _, ww, hh = box
            if ww * hh < self.min_area:
                continue
            if self.max_ratio and (ww / hh > self.max_ratio or
                                   hh / ww > self.max_ratio):
                continue
            candidates.append(box)
        # print('end with', len(candidates))
        candidates = np.array(candidates)
        if self.max_boxes:
            candidates = candidates[:self.max_boxes]

        # convert width, height to xmax, ymax
        candidates[:, 2] += candidates[:, 0]
        candidates[:, 3] += candidates[:, 1]

        # return the boxes and None in place of a list of masks
        return candidates


class FeatureExtractor:
    def __init__(self, imageInputSize=None):
        self.imageInputSize = imageInputSize

    def preprocessImage(self, image):
        if type(image) is str:
            # load the image from the given path
            image = cv2.imread(image)

        if self.imageInputSize:
            # resize the image to be the input size on its shorter side
            image = mycv.scaleImage(image, self.imageInputSize)
        return image

    def detectAndCompute(self, image, mask=None, **kwargs):
        """
        Subclasses must implement this function.

        Finds a set of feature points in the image, and descriptors for them.

        Actually, this just wraps the detectAndComputeBoxes function,
        ignoring the image it returns, returning the centers of the boxes as
        points and flattened descriptor tensors as descriptor vectors.  This \
        function is provided to comply with opencv's usual point-and-flat-\
        descriptor data scheme.

        Arguments:
        image -- An image or path to an image file.
        mask -- A binary mask indicating which pixels to ignore (not used for\
        now).

        Return values:
        - a list of feature points (centers of bounding boxes)
        - a list of feature vectors (flattened feature tensors)
        """
        # This method is only here to provide an "interface", and should never
        # actually be called.  If it is, raise an error.
        raise NotImplementedError()

    @property
    def device(self):
        return torch.device("cpu")


class ObjectExtractor(FeatureExtractor):

    def __init__(self, aggBoxFunction=None, iouThreshold=None,
                 filterSmall=False, filterAtEdges=False, imageInputSize=None,
                 siftSubFeatures=False, siftSubParams={}, d2SubFeatures=False,
                 d2SubParams={}):
        super().__init__(imageInputSize=imageInputSize)
        self.aggBoxFunction = aggBoxFunction
        self.iouThreshold = iouThreshold
        self.filterSmall = filterSmall
        self.filterAtEdges = filterAtEdges
        self.siftSubFeatures = siftSubFeatures
        self.siftSubParams = siftSubParams
        if d2SubFeatures:
            if siftSubFeatures:
                raise ValueError("Cannot have both SIFT and D2 sub-features!")
            self.d2Reader = D2NetReader(**d2SubParams)
        else:
            self.d2Reader = None



    def detectAndCompute(self, image, mask=None, **kwargs):
        """
        Finds a set of feature points in the image, and descriptors for them.

        Actually, this just wraps the detectAndComputeBoxes function,
        ignoring the image it returns, returning the centers of the boxes as
        points and flattened descriptor tensors as descriptor vectors.  This \
        function is provided to comply with opencv's usual point-and-flat-\
        descriptor data scheme.

        Arguments:
        image -- An image or path to an image file.
        mask -- A binary mask indicating which pixels to ignore (not used for\
        now).

        Return values:
        - a list of feature points (centers of bounding boxes)
        - a list of feature vectors (flattened feature tensors)
        """
        boxes, descs, _ = self.detectAndComputeBoxes(image, **kwargs)
        points = [geo.getCenterFromBox(box[:4]) for box in boxes]
        vecs = [d.flatten() for d in descs]
        return points, vecs

    def detectAndComputeBoxes(self, image, **kwargs):
        """
        Subclasses must implement this function.

        Extracts bounding boxes and feature vectors of objects in an image.

        Return values:
        - a list of object bounding boxes, in [xmin ymin xmax ymax] format
        - a 4D numpy array of object feature tensors, indexed by object along \
        the most major axis
        """
        image = self.preprocessImage(image)
        boxes, masks, descriptors = self._detectAndComputeBoxes(image, **kwargs)

        if self.filterSmall:
            # num of pixels per dim encompassed by one conv5 pixel
            limit = 16 * 7
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            bigEnoughIdxs = np.where(np.logical_and(ws > limit, hs > limit))[0]
            boxes = boxes[bigEnoughIdxs]
            descriptors = descriptors[bigEnoughIdxs]

        if self.filterAtEdges:
            insideLeft = boxes[:, 0] > 0
            insideTop = boxes[:, 1] > 0
            insideRight = boxes[:, 2] < image.shape[1] - 1
            insideBottom = boxes[:, 3] < image.shape[0] - 1
            insideX = np.logical_and(insideLeft, insideRight)
            insideY = np.logical_and(insideTop, insideBottom)
            insideIdxs = np.where(np.logical_and(insideX, insideY))[0]
            boxes = boxes[insideIdxs]
            descriptors = [descriptors[i] for i in insideIdxs]

        return boxes, masks, descriptors, image

    def detectAndComputeObjects(self, image, **kwargs):
        scene = self.detectAndComputeScene(image, **kwargs)
        return scene.objects, scene.image

    def detectAndComputeScene(self, image, keep_preprocced_img=True, **kwargs):
        boxes, masks, descriptors, preprocessedImg = \
            self.detectAndComputeBoxes(image, **kwargs)
        subFeats = None

        if self.d2Reader or self.siftSubFeatures:
            if (boxes is None or len(boxes) == 0) and len(subFeats) > 0:
                # there are valid subfeatures, so make a single box
                # encompassing the whole image
                img_h, img_w, _ = preprocessedImg.shape
                boxes = np.array([(0, 0) + (img_w, img_h)])
                descriptors = self.compute(preprocessedImg, boxes)

        if self.d2Reader:
            # we need the path to the image!
            assert(type(image) in [Path, str])
            kps, descs = self.d2Reader.read_cached_features(image)
            # convert keypoints to opencv keypoint objects
            # 'size' is a dummy value here, we don't use it anywhere
            kps = [cv2.KeyPoint(*rr, _size=1) for rr in kps]
            subFeats = list(zip(kps, descs))

        elif self.siftSubFeatures:
            subFeats = self.getSiftSubFeatures(preprocessedImg)

        if not keep_preprocced_img:
            preprocessedImg = None
        scene = Scene(boxes, descriptors, preprocessedImg, masks, subFeats)
        return scene

    def getObjectsFromBoxes(self, image, boxes, masks=None):
        scene = self.getSceneFromBoxes(image, boxes, masks)
        return scene.objects, scene.image

    def getSceneFromBoxes(self, image, boxes, masks=None):
        preprocessedImg = self.preprocessImage(image)
        descriptors = self.compute(preprocessedImg, np.array(boxes))
        if masks is None:
            masks = [None] * len(boxes)
        subFeats = None
        if self.siftSubFeatures:
            subFeats = self.getSiftSubFeatures(preprocessedImg)
        return Scene(preprocessedImg, boxes, descriptors, masks, subFeats)

    def getSiftSubFeatures(self, image):
        siftEngine = cv2.xfeatures2d.SIFT_create(**self.siftSubParams)
        siftKps, descs = siftEngine.detectAndCompute(image, None)
        if len(siftKps) == 0:
            return []
        allFeatures = list(zip(siftKps, descs))
        return allFeatures

    def setSiftSubFeatures(self, image, objects):
        allFeatures = self.getSiftSubFeatures(image)
        pts = np.array([kp.pt for kp, _ in allFeatures])
        if len(pts) > 0:
            for ob in objects:
                insideHoriz = np.logical_and(pts[:, 0] > ob.box[0],
                                             pts[:, 0] < ob.box[2])
                insideVert = np.logical_and(pts[:, 1] > ob.box[1],
                                            pts[:, 1] < ob.box[3])
                insideMask = np.logical_and(insideHoriz, insideVert)
                objSubPtsMask = insideMask
                objSubIdxs = np.where(objSubPtsMask)[0]
                ob.setSubFeatures(objSubIdxs, allFeatures)

    def aggregateBoxes(self, boxes):
        if self.aggBoxFunction is None:
            return boxes

        if self.iouThreshold is None:
            print(self.aggBoxFunction.__name__,
                  'provided to aggregate boxes, but no threshold set!')

        else:
            if boxes.shape[1] > 4:
                scores = boxes[:, 4]
            else:
                scores = None
            boxes, _ = self.aggBoxFunction(boxes, self.iouThreshold, scores)

        return boxes

    def setAggBoxFunction(self, aggBoxFunction):
        self.aggBoxFunction = aggBoxFunction

    def setIouThreshold(self, thresh):
        self.iouThreshold = thresh


class SeparateExtractorMixin(object):
    def _detectAndComputeBoxes(self, image):
        boxes, masks = self.detect(image)
        descs = self.compute(image, boxes)
        return boxes, masks, descs

    def detect(self, image):
        boxes, masks = self.boxFunction(image)
        boxes = self.aggregateBoxes(boxes)
        # print('final box count:', len(boxes))
        return boxes, masks[:len(boxes)]


class FixedCnnPatchExtractor(ObjectExtractor, SeparateExtractorMixin):
    def __init__(self, netName, patchSize, feature, compression=None,
                 ss_params=None, eb_params=None, device=None, **kwargs):
        super().__init__(**kwargs)
        self.featureBlobName = feature
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patchSize, patchSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        # load model and discard classification layer
        if netName in RESNET_KEYS:
            self.net = get_resnet_feat_extractor(patchSize, feature, netName)
        elif netName == ALEX_KEY:
            self.net = get_alexnet_feat_extractor(feature)
        elif netName in VGG_KEYS:
            self.net = get_vgg_feat_extractor(patchSize, feature, netName)
        elif netName in DENSE_KEYS:
            self.net = get_densenet_feat_extractor(patchSize, feature,
                                                   netName, compression)

        # disable gradients and set to eval, since we're not learning here
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        # move model to the GPU if it is available
        if not device:
            self._device = torch.device("cuda:0" if torch.cuda.is_available()
                                         else "cpu")
        else:
            self._device = device
        self.net = self.net.to(self.device)

        if ss_params:
            self.box_engine = SSExtractor(**ss_params)
        elif eb_params:
            self.box_engine = EdgeBoxExtractor(**eb_params)

    def boxFunction(self, image):
        boxes = self.box_engine.get_boxes(image)
        if len(boxes) > 0:
            boxes = self.aggregateBoxes(boxes)
        return boxes, [None] * len(boxes)

    def compute(self, image, boxes, batch_size=512):
        if len(boxes) == 0:
            return np.zeros((0, 0))

        # opencv images come in BGR channel order, but pytorch expects RGB
        pytorch_image = np.flip(image, axis=2)

        # extract the patches and format them for pytorch
        patches = [pytorch_image[b[1]:b[3], b[0]:b[2], :]
                   for b in boxes.astype(int) if b[3] > b[1] and b[2] > b[0]]
        # transform it to a pytorch-compatible image
        patches = torch.stack([self.transform(patch) for patch in patches])

        batch_descs = []
        batches = torch.split(patches, batch_size)
        for batch in batches:
            desc_batch = self.net(batch.to(self.device))
            # print('size:', desc_batch.size // desc_batch.shape[0])
            batch_descs.append(desc_batch.reshape(len(batch), -1))
        return np.concatenate(batch_descs)

    @property
    def device(self):
        return self._device


class LiftWrapperExtractor(ObjectExtractor):
    def __init__(self, liftDir, imageInputSize=None):
        super().__init__(imageInputSize=imageInputSize)
        self.liftDir = liftDir

    def _detectAndComputeBoxes(self, image):
        kps, descs = extractLiftFeatures(self.liftDir, image)
        boxes = np.zeros((len(kps), 4), dtype=np.float32)
        boxes[:, 0] = kps[:, 0] - 1
        boxes[:, 1] = kps[:, 1] - 1
        boxes[:, 2] = kps[:, 0] + 1
        boxes[:, 3] = kps[:, 1] + 1
        return boxes, None, descs


class SiftWrapperExtractor(ObjectExtractor):
    def __init__(self, siftKwargs={}, imageInputSize=None):
        super().__init__(imageInputSize=imageInputSize)
        self.setSiftKwargs(siftKwargs)

    def setSiftKwargs(self, siftKwargs):
        self.siftEngine = cv2.xfeatures2d.SIFT_create(**siftKwargs)

    def _detectAndComputeBoxes(self, image, mask=None, **kwargs):
        kps, descriptors = self.siftEngine.detectAndCompute(image, None)
        points = [kp.pt for kp in kps]
        boxes = points_to_boxes(points)
        if descriptors is None:
            descriptors = [None] * len(boxes)
        # the middle element is a placeholder for masks
        return boxes, None, descriptors


class OrbWrapperExtractor(ObjectExtractor):
    def __init__(self, orbKwargs={}, imageInputSize=None):
        super().__init__(
            imageInputSize=imageInputSize)
        self.setOrbKwargs(orbKwargs)

    def setOrbKwargs(self, orbKwargs):
        self.orbEngine = cv2.ORB_create(**orbKwargs)

    def _detectAndComputeBoxes(self, image, mask=None, **kwargs):
        kps, descriptors = self.orbEngine.detectAndCompute(image, None)
        points = [kp.pt for kp in kps]
        boxes = points_to_boxes(points)
        if descriptors is None:
            descriptors = [None] * len(boxes)
        # the middle element is a placeholder for masks
        return boxes, [None]*len(boxes), descriptors


class CachedFeatureReader:
    def __init__(self, feat_ext, **kwargs):
        if not feat_ext.startswith('.'):
            feat_ext = '.' + feat_ext
        self.feat_ext = feat_ext

    def read_cached_features_as_objects(self, image_path):
        # perform no filtering based on the scores
        scene = self.read_cached_features_as_scene(image_path)
        return scene.objects

    def read_cached_features_as_scene(self, image_path):
        points, descs = self.read_cached_features(image_path)
        boxes = points_to_boxes(points)
        return Scene(boxes, descs)

    def read_cached_features(self, image_path):
        raise NotImplementedError("subclasses must implement this!")


class D2NetReader(CachedFeatureReader):
    def __init__(self, max_features):
        super().__init__('.d2-net')
        self.max_features = max_features
        print('initialized with', self.max_features)

    def read_cached_features(self, image_path):
        # perform no filtering based on the scores
        feature_path = image_path + self.feat_ext
        stored = np.load(feature_path)
        zp = zip(stored['keypoints'], stored['scores'], stored['descriptors'])
        kept = sorted(zp, key=lambda xx: xx[1],
                      reverse=True)[:self.max_features]
        kps, scores, descs = zip(*kept)
        kps = np.array(kps)[:, :2]

        return kps, descs


class RandomExtractor(ObjectExtractor):
    def _detectAndComputeBoxes(self, image, **kwargs):
        # generate random boxes
        # generate random features for boxes
        N = 100
        D = 128
        scales = (image.shape[1::-1]) * 2
        boxes = (np.random.random((N, 4)) * scales).astype(int)
        descriptors = np.random.random((N, D))
        # the middle element is a placeholder for masks
        return boxes, [None]*len(boxes), descriptors


def points_to_boxes(points):
    return np.array([(p[0]-1, p[1]-1, p[0]+1, p[1]+1) for p in points],
                     dtype=np.float32)


def extractLiftFeatures(lift_dir, image):
    # save the image to a temporary file
    imgfile = tempfile.NamedTemporaryFile(suffix='.png')
    cv2.imwrite(imgfile.name, image)
    cwd = os.getcwd()
    os.chdir(lift_dir)
    kpfile = tempfile.NamedTemporaryFile(suffix='.txt')
    h5file = tempfile.NamedTemporaryFile(suffix='.h5')
    # format the commands to invoke LIFT
    cmd = "python main.py --task=test --subtask=kp --logdir=logs/test --test_img_file={imgpath} \
    --test_out_file={kptxt} --pretrained_kp=release-aug/kp/ --use_batch_norm=False \
    --mean_std_type=hardcoded && \
    python main.py --task=test --subtask=ori --logdir=logs/test \
    --test_img_file={imgpath} --test_out_file={kptxt} --test_kp_file={kptxt} \
    --pretrained_ori=release-aug/ori/ --use_batch_norm=False --mean_std_type=hardcoded && \
    python main.py --task=test --subtask=desc --logdir=logs/test --test_img_file={imgpath} \
    --test_out_file={final} --test_kp_file={kptxt} --pretrained_desc=release-aug/desc/ \
    --use_batch_norm=False --mean_std_type=hardcoded"
    fmtd_cmd = cmd.format(imgpath=imgfile.name, kptxt=kpfile.name,
                          final=h5file.name)
    # run the commands, suppressing regular output
    subprocess.call([fmtd_cmd], shell=True, stdout=subprocess.DEVNULL)
    # read the keypoints and features from the resulting .h5 file, return them
    data = h5py.File(h5file.name)
    # first two dimensions correspond to x and y coordinates
    kps = data['keypoints'].value[:, :2]
    descs = data['descriptors'].value
    os.chdir(cwd)
    imgfile.close()
    kpfile.close()
    h5file.close()
    return kps, descs


def dictOfListsToListOfDicts(dictOfLists):
    """
    Arguments:
    -- dictOfLists: a dictionary of lists all of the same length
    Returns:
    -- listOfDicts: a list where listOfDicts[i][j] == dictOfLists[j][i]
    """
    listOfDicts = [{} for _ in list(dictOfLists.values())[0]]
    for key in dictOfLists:
        for d, elem in zip(listOfDicts, dictOfLists[key]):
            d[key] = elem
    return listOfDicts


# Functions for filtering bounding boxes


# The below method is that of Malisiewicz et al., modified slightly.
# originally found here:
# http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def collectOverlaps(boxes, iouThresh, scores=None):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indices
    groups = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box, or by the
    # scores of the boxes if they are provided
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the intersection over union
        intersectionSize = w * h
        unionSize = area[idxs[:last]] + area[i] - intersectionSize
        iou = intersectionSize / unionSize

        # delete all indexes from the index list that have
        overlapIdxIdxs = np.concatenate(([last], np.where(iou > iouThresh)[0]))
        groups.append(idxs[overlapIdxIdxs])
        idxs = np.delete(idxs, overlapIdxIdxs)

    # return a list of lists of indices indicating the groups
    return groups


def encompassBoxesFast(boxes, iouThresh, scores=None):
    if len(boxes) == 0:
        return boxes, []
    groups = collectOverlaps(boxes, iouThresh, scores)
    containingBoxes = []
    for group in groups:
        if scores is not None:
            weights = scores[group].flatten()
        else:
            weights = np.ones(len(group))
        containingBox = np.zeros(4)
        containingBox[:2] = np.min(boxes[group, :2], axis=0)
        containingBox[2:4] = np.max(boxes[group, 2:4], axis=0)
        containingBoxes.append(np.append(containingBox, np.sum(weights)))

    return np.array(containingBoxes), groups


def averageBoxesFast(boxes, iouThresh, scores=None):
    if len(boxes) == 0:
        return boxes, []
    groups = collectOverlaps(boxes, iouThresh, scores)
    avgBoxes = []
    for group in groups:
        if scores is not None:
            weights = scores[group].flatten()
        else:
            weights = np.ones(len(group))
        avgBox = np.average(boxes[group, :4], axis=0, weights=weights)
        avgBoxes.append(np.append(avgBox, np.sum(weights)))

    avgBoxes = np.array(avgBoxes)
    # sort in descending order by score
    sortedAvgBoxes = avgBoxes[avgBoxes[:, -1].argsort()[::-1]]
    return sortedAvgBoxes, groups


def nonMaxSuppressionFast(boxes, iouThresh, scores=None, returnIndices=False):
    if len(boxes) == 0:
        return boxes, []

    groups = collectOverlaps(boxes, iouThresh, scores)

    maxIdxs = []
    for group in groups:
        if scores is not None:
            groupScores = scores[group]
            groupMaxScoreIdx = np.argmax(groupScores)
        else:
            # just pick the first one
            groupMaxScoreIdx = 0
        idxOfmaxBoxInGroup = group[groupMaxScoreIdx]
        maxIdxs.append(idxOfmaxBoxInGroup)
    if returnIndices:
        return maxIdxs, groups
    else:
        return boxes[maxIdxs], groups


def visualizeObjectness(imagePath, boxes, normalize=True):
    image = cv2.imread(imagePath)
    objectness = np.zeros(image.shape, dtype=float)
    for box in boxes.astype(int):
        xmin, ymin, xmax, ymax = box[:4]
        objectness[xmin:xmax, ymin:ymax, :] += 1

    if normalize:
        objectness -= objectness.min()
        objectness /= objectness.max()
        objectness *= 255

    dispObj = objectness.astype(np.uint8)
    pair = mycv.getDoubleImage(image, dispObj)
    cv2.imshow('objectness', pair)
    cv2.waitKey(0)


def sizeFunction(boxes):
    return np.sqrt((boxes[:, 2] - boxes[:, 0] + 1) \
                   * (boxes[:, 3] - boxes[:, 1] + 1))


def ratioFunction(boxes):
    ws = (boxes[:, 2] - boxes[:, 0] + 1).astype(float)
    hs = (boxes[:, 3] - boxes[:, 1] + 1).astype(float)
    return np.max((ws / hs, hs / ws), axis=0)
