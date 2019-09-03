import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnet50
from torchvision.ops import MultiScaleRoIAlign


class TwoMLPHead(nn.Module):
    """
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6_cls = nn.Linear(in_channels, representation_size)
        self.fc6_dropout_cls = nn.Dropout(0.5)
        self.fc7_cls = nn.Linear(representation_size, representation_size)
        self.fc7_dropout_cls = nn.Dropout(0.5)

        self.fc6_reg = nn.Linear(in_channels, representation_size)
        self.fc6_dropout_reg = nn.Dropout(0.5)
        self.fc7_reg = nn.Linear(representation_size, representation_size)
        self.fc7_dropout_reg = nn.Dropout(0.5)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x_cls = F.relu(self.fc6_cls(x))
        x_cls = self.fc6_dropout_cls(x_cls)
        x_cls = F.relu(self.fc7_cls(x_cls))
        x_cls = self.fc7_dropout_cls(x_cls)

        x_reg = F.relu(self.fc6_reg(x))
        x_reg = self.fc6_dropout_reg(x_reg)
        x_reg = F.relu(self.fc7_reg(x_reg))
        x_reg = self.fc7_dropout_reg(x_reg)

        return x_cls, x_reg


class RCNNPredictor(nn.Module):
    """
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(RCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x_cls, x_reg = x
        scores = self.cls_score(x_cls)
        bbox_deltas = self.bbox_pred(x_reg)

        return scores, bbox_deltas


def get_model_instance(num_classes):
    resent50 = resnet50(pretrained=False)
    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = resent50.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    backbone = BackboneWithFPN(resent50, return_layers, in_channels_list, out_channels)
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=[0, 1, 2, 3],
        output_size=7,
        sampling_ratio=2)

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(
        out_channels * resolution ** 2,
        representation_size)

    representation_size = 1024
    box_predictor = RCNNPredictor(
        representation_size,
        num_classes)
    model = FasterRCNN(backbone, image_mean=[0.5, 0.5, 0.5], image_std=[1, 1, 1],
                       box_roi_pool=box_roi_pool, box_head=box_head, box_predictor=box_predictor)

    return model
