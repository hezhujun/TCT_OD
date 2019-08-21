import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import BackboneWithFPN

import transforms as T
import utils
from dataset_voc import TCTDataset
from engine import train_one_epoch, evaluate


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


label2id = {
    "adenocarcinoma": 1,
    "agc": 2,
    "asch": 3,
    "ascus": 4,
    "dysbacteriosis": 5,
    "hsil": 6,
    "lsil": 7,
    "monilia": 8,
    "normal": 9,
    "vaginalis": 10
}


def label_transform(label):
    id = label2id.get(label, -1)
    if id == -1:
        raise ValueError("lable \"{}\" is unknown".format(label))
    return id


def get_model_instance(num_classes):
    resent50 = torchvision.models.resnet50(pretrained=False)
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
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes, image_mean=[0.5, 0.5, 0.5],
                                                    image_std=[1, 1, 1])
    return model


def main():
    num_classes = len(label2id) + 1
    model = get_model_instance(num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device : {}".format(device.type))
    model.to(device)

    print("Loading data")
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET", transform=get_transform(train=True),
                         label_transform=label_transform, image_set="train")
    dataset_val = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET",
                             transform=get_transform(train=False), label_transform=label_transform, image_set="val")
    dataset_test = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET",
                              transform=get_transform(train=False), label_transform=label_transform, image_set="test")

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        torch.save(model.state_dict(), "model_{}.pth".format(epoch))
        evaluate(model, data_loader_val, device)

    evaluate(model, data_loader_test, device)


if __name__ == '__main__':
    main()
