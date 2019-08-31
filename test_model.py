import argparse
import copy
import datetime
import time
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.backbone_utils import BackboneWithFPN

import coco_eval
import coco_utils
import transforms as T
import utils
from _utils.log_utils import Log
from _utils.model_path_manager import ModelPathManager
from dataset import TCTDataset
import engine

from tct_train import *
from result_recorder import ResultRecorder


def test_model(args):
    dataset = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/train.json",
        get_transforms(False))
    dataset_val = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/val.json",
        get_transforms(False))
    dataset_test = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/test.json",
        get_transforms(False))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  collate_fn=utils.collate_fn)

    coco_api = coco_utils.get_coco_api_from_dataset(dataset_val)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model = get_model_instance(num_classes=len(coco_api.cats) + 1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: {}".format(device.type))

    print("loading model from", args.pretrain_model)
    checkpoint = torch.load(args.pretrain_model)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    rr = ResultRecorder("dataset/tct_result_train.json")
    evaluate(model, data_loader, device, None, None, head="Train:")
    rr.save()
    # rr = ResultRecorder("dataset/tct_result_val.json")
    # evaluate(model, data_loader_val, device, None, None, head="Eval:", result_recorder=rr)
    # rr.save()
    rr = ResultRecorder("dataset/tct_result_test.json")
    evaluate(model, data_loader_test, device, None, None, head="Test :")
    rr.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lr-steps", default=[30, ], nargs='+', type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--pretrain-model")
    args = parser.parse_args()
    test_model(args)
