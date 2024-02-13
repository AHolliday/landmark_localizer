import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models import AlexNet
from torchvision.models import vgg as torch_vgg_models
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.densenet import DenseNet, _Transition


# custom pytorch classes to extract intermediate features from stock CNNs


class AlexNetFeatureExtractor(AlexNet):
    def __init__(self, feature_to_use, input_size=64, **kwargs):
        super(AlexNetFeatureExtractor, self).__init__(**kwargs)
        self.feature_to_use = feature_to_use
        self.input_size = input_size

    def forward(self, act):
        # features are conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5
        conv_count = 0
        for layer in self.features:
            act = layer(act)
            if type(layer) is nn.Conv2d:
                conv_count += 1
                name = 'conv' + str(conv_count)
            elif type(layer) is nn.MaxPool2d:
                name = 'pool' + str(conv_count)
            if name == self.feature_to_use:
                return act.cpu().numpy()

    def get_layer_shapes(self):
        layer_shapes = {}
        dummy_data = torch.zeros((1, 3, self.input_size, self.input_size))
        act = dummy_data
        conv_count = 0
        for layer in self.features:
            act = layer(act)
            if type(layer) is nn.Conv2d:
                conv_count += 1
                name = 'conv' + str(conv_count)
            elif type(layer) is nn.MaxPool2d:
                name = 'pool' + str(conv_count)
            layer_shapes[name] = act.shape
        return layer_shapes



def get_alexnet_feat_extractor(feature_to_use, pretrained=True, **kwargs):
    model = AlexNetFeatureExtractor(feature_to_use, **kwargs)
    if pretrained:
        base_model = models.alexnet(pretrained, **kwargs)
        model.load_state_dict(base_model.state_dict(), strict=False)
    return model


class DenseNetFeatureExtractor(DenseNet):
    @classmethod
    def from_densenet(cls, densenet_object, *args, **kwargs):
        densenet_object.__class__ = cls
        densenet_object._init_helper(*args, **kwargs)
        return densenet_object

    def __init__(self, input_size, feature_to_use, compression=None,
                 **kwargs):
        super(DenseNetFeatureExtractor, self).__init__(**kwargs)
        self._init_helper(input_size, feature_to_use, compression, **kwargs)

    def _init_helper(self, input_size, feature_to_use=None, compression=None,
                     **kwargs):
        self.feature_to_use = feature_to_use
        self.input_size = input_size
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for param in self.parameters():
            param.requires_grad = False

        self.compressor = None
        if compression:
            self.input_size = input_size
            test_input = torch.zeros((1, 3, input_size, input_size),
                                     requires_grad=False)
            feat = self.forward(test_input)

            self.compressor = get_linear_compressor(feat.size, compression)

    def forward(self, input_data):
        act = input_data
        for feat_name, layer in self.features._modules.items():
            act = layer(act)
            if feat_name == self.feature_to_use:
                if self.compressor:
                    # flatten non_batch dimensions, then compress
                    act = self.compressor(act.view(act.shape[0], -1))
                return act.cpu().numpy()

        # if we get here, we never found a feature to return.  Uh-oh!
        raise ValueError('feature', self.feature_to_use, 'not in network!')

    def get_layer_shapes(self):
        layer_shapes = {}
        dummy_data = torch.zeros((1, 3, self.input_size, self.input_size))
        act = dummy_data
        for feat_name, layer in self.features._modules.items():
            act = layer(act)
            layer_shapes[feat_name] = act.shape
        return layer_shapes


def get_densenet_feat_extractor(input_size, feature_to_use, type,
                                compression=None, pretrained=True, **kwargs):
    if type == 'densenet121':
        func = models.densenet.densenet121
    elif type == 'densenet161':
        func = models.densenet.densenet161
    elif type == 'densenet169':
        func = models.densenet.densenet169
    elif type == 'densenet201':
        func = models.densenet.densenet201
    # TODO
    # if we have a cached version of our own and pretrained=True, load it
    # else...
    base_model = func(pretrained=pretrained, **kwargs)
    model = DenseNetFeatureExtractor.from_densenet(
        base_model, input_size, feature_to_use, compression, **kwargs)
    return model


class ResNetFeatureExtractor(ResNet):
    def __init__(self, block, layers, input_size, feature_to_use):
        """
        Partially copied from the constructor of ResNet, as some changes in
        the "inside" of how it worked needed to be made.

        block: The class (BasicBlock or Bottleneck) to be used for the net's
            internal blocks.
        layers: a list of integers indicating how many layers will belong to
            each block.
        input_size: pixel height and width of the square input patches."""
        self.feature_to_use = feature_to_use
        self.input_size = input_size
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        stride = 2
        # if stride == 2 and input_size <= 32:
        #     stride = 1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride)
        # if stride == 2 and input_size <= 64:
        #     stride = 1
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride)
        # if stride == 2 and input_size <= 128:
        #     stride = 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                              nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        intermediate = self.conv1(input_data)
        intermediate = self.bn1(intermediate)
        intermediate = self.relu(intermediate)
        pool1 = self.maxpool(intermediate)
        if self.feature_to_use == 'pool1':
            return pool1.detach().cpu().numpy()
        res2c = self.layer1(pool1)
        if self.feature_to_use == 'res2c':
            return res2c.detach().cpu().numpy()
        res3d = self.layer2(res2c)
        if self.feature_to_use == 'res3d':
            return res3d.detach().cpu().numpy()
        res4f = self.layer3(res3d)
        if self.feature_to_use == 'res4f':
            return res4f.detach().cpu().numpy()
        res5c = self.layer4(res4f)
        if self.feature_to_use == 'res5c':
            return res5c.detach().cpu().numpy()
        pool5 = self.avgpool(res5c)
        if self.feature_to_use == 'pool5':
            return pool5.detach().cpu().numpy()

    def get_layer_shapes(self):
        layer_shapes = {}
        dummy_data = torch.zeros((1, 3, self.input_size, self.input_size))
        intermediate = self.conv1(dummy_data)
        intermediate = self.bn1(intermediate)
        intermediate = self.relu(intermediate)
        pool1 = self.maxpool(intermediate)
        layer_shapes['pool1'] = pool1.shape
        res2c = self.layer1(pool1)
        layer_shapes['res2c'] = res2c.shape
        res3d = self.layer2(res2c)
        layer_shapes['res3d'] = res3d.shape
        res4f = self.layer3(res3d)
        layer_shapes['res4f'] = res4f.shape
        res5c = self.layer4(res4f)
        layer_shapes['res5c'] = res5c.shape
        try:
            pool5 = self.avgpool(res5c)
            layer_shapes['pool5'] = pool5.shape
        except RuntimeError:
            print('Input size', self.input_size,
                  'does not allow resnet extractor to compute pool5.')
        return layer_shapes


def get_resnet_feat_extractor(input_size, feature_to_use, type,
                              pretrained=True, **kwargs):
    if type == 'resnet18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
        func = models.resnet.resnet18
    elif type == 'resnet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
        func = models.resnet.resnet34
    elif type == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
        func = models.resnet.resnet50
    elif type == 'resnet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
        func = models.resnet.resnet101
    elif type == 'resnet152':
        block = Bottleneck
        layers = [3, 8, 36, 3]
        func = models.resnet.resnet152

    model = ResNetFeatureExtractor(block, layers, input_size, feature_to_use,
                                   **kwargs)
    if pretrained:
        base_weights = func(True, **kwargs).state_dict()
        # remove fc weights
        base_weights = {kk: base_weights[kk] for kk in base_weights
                        if not kk.startswith('fc')}
        model.load_state_dict(base_weights)
    return model


# Custom VGG classes


class VGGFeatureExtractor(torch_vgg_models.VGG):
    def __init__(self, backbone, input_size, feature_to_use):
        """
        backbone: a torch.nn.Sequential containing the network backbone
        input_size: int giving the width (and height, as we assume square) of
            the input
        feature_to_use: name of the feature to extract
        """
        super(VGGFeatureExtractor, self).__init__(backbone)
        self.out_feature_layer = None
        self.input_size = input_size

        # define the locations and names of each feature layer
        pool_count = 0
        for ii, layer in enumerate(self.features):
            # use every pooling layer, and the layer just before it, as a feature.
            if type(layer) is nn.Conv2d or type(layer) is nn.BatchNorm2d:
                last_feat_idx = ii

            elif type(layer) is nn.MaxPool2d:
                pool_count += 1
                key = 'pool{}'.format(pool_count)
                pre_key = 'pre_' + key
                if pre_key == feature_to_use:
                    self.out_feature_layer = last_feat_idx
                if key == feature_to_use:
                    self.out_feature_layer = ii
        if self.out_feature_layer is None:
            raise ValueError(feature_to_use + ' is not a valid feature!')

    def forward(self, input_data):
        activation = input_data
        for ii, layer in enumerate(self.features):
            activation = layer(activation)
            if ii == self.out_feature_layer:
                return activation.cpu().numpy()

    def get_layer_shapes(self):
        layer_shapes = {}
        dummy_data = torch.zeros((1, 3, self.input_size, self.input_size))
        act = dummy_data
        pool_count = 0
        for ii, layer in enumerate(self.features):
            act = layer(act)
            if type(layer) is nn.Conv2d:
                last_conv_size = act.shape
            if type(layer) is nn.MaxPool2d:
                pool_count += 1
                layer_shapes['pre_pool{}'.format(pool_count)] = last_conv_size
                layer_shapes['pool{}'.format(pool_count)] = act.shape
        return layer_shapes


def get_vgg_feat_extractor(input_size, feature_to_use, type,
                             pretrained=True, **kwargs):
    if type == 'vgg11':
        backbone = torch_vgg_models.cfg['A']
        weight_func = models.vgg.vgg11_bn
    elif type == 'vgg13':
        backbone = torch_vgg_models.cfg['B']
        weight_func = models.vgg.vgg13_bn
    elif type == 'vgg16':
        backbone = torch_vgg_models.cfg['D']
        weight_func = models.vgg.vgg16_bn
    elif type == 'vgg19':
        backbone = torch_vgg_models.cfg['E']
        weight_func = models.vgg.vgg19_bn
    layers = torch_vgg_models.make_layers(backbone, batch_norm=True)
    model = VGGFeatureExtractor(layers, input_size, feature_to_use, **kwargs)
    if pretrained:
        base_weights = weight_func(pretrained, **kwargs).state_dict()
        model.load_state_dict(base_weights, strict=False)
    return model


# class InceptionFeatureExtractor:
#     """Based on Inception V3 as provided in torchvision"""



def get_linear_compressor(in_size, factor=16, mean=0, std=1):
    """A la Sunderhauf 2015, we use a Gaussian compression matrix."""
    out_size = in_size // factor
    print('compressor size:', in_size * out_size * 4 / (10**6), 'MB')
    # make sure we always get the same compressor
    torch.manual_seed(0)
    compressor = nn.Linear(in_size, out_size, bias=False)
    matrix = torch.normal(torch.ones(out_size, in_size) * mean, std)
    compressor.weight.data = matrix
    return compressor


if __name__ == "__main__":
    model = torchvision.models.vgg.vgg16_bn(pretrained=True)
    print(model)
    import pdb; pdb.set_trace()
