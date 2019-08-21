import argparse
import copy
import datetime
import time

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


def get_transforms(train):
    transforms = []
    transforms.append(T.TargetToTensor())
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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


def train_one_epoch(model, optimizer, data_loader, device, epoch, log, head="", print_freq=1):
    space_fmt = ':0' + str(len(str(len(data_loader)))) + 'd'
    if torch.cuda.is_available() and (torch.device("cuda") == device):
        log_msg = "  ".join([
            head,
            "[{iter" + space_fmt + "}/{total_iter}]",
            "eta: {eta}",
            "{metric}",
            "time: {time:.4f}",
            "data: {data:.4f}",
            "max mem: {memory:.0f}"
        ])
    else:
        log_msg = "  ".join([
            head,
            "[{iter" + space_fmt + "}/{total_iter}]",
            "eta: {eta}",
            "{metric}",
            "time: {time:.4f}",
            "data: {data:.4f}"
        ])
    MB = 1024.0 * 1024.0

    model.train()

    start_time = time.time()
    end = time.time()
    i = 0
    for images, targets in data_loader:
        data_time = time.time() - end
        log.log("data_time", data_time)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_item = losses.item()
        log.log("loss", loss_item)
        loss_dict_item = {"loss": loss_item}
        for k, v in loss_dict.items():
            loss_dict_item[k] = v.item()
            log.log(k, v.item())
        metric_str = []
        for key, value in loss_dict_item.items():
            metric_str.append("{}: {:.4f}".format(key, value))
        metric_str = "  ".join(metric_str)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        iter_time = time.time() - end
        log.log("iter_time", iter_time)
        log.commit(epoch=epoch, iteration=(epoch * len(data_loader) + i))
        if i % print_freq == 0 or i == len(data_loader) - 1:
            eta_seconds = iter_time * (len(data_loader) - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if torch.cuda.is_available() and (torch.device("cuda") == device):
                print(
                    log_msg.format(iter=i, total_iter=len(data_loader), eta=eta_string, metric=metric_str,
                                   time=iter_time, data=data_time,
                                   memory=(torch.cuda.max_memory_allocated() / MB)))
            else:
                print(
                    log_msg.format(iter=i, total_iter=len(data_loader), eta=eta_string, metric=metric_str,
                                   time=iter_time, data=data_time))
        i += 1
        end = time.time()

    total_time = time.time() - start_time
    log.log("total_time", total_time)
    log.commit(epoch=epoch)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("{} Total time: {} ({:.4f}s / it)".format(head, total_time_str, total_time / len(data_loader)))


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=None, log=None, head="", print_freq=1):
    space_fmt = ":0" + str(len(str(len(data_loader)))) + "d"
    log_msg = "  ".join([
        head,
        "[{iter" + space_fmt + "}/{total_iter}]",
        "eta: {eta}",
        "model_time: {model_time:.4f}",
        "evaluator_time: {eval_time:.4f}",
        "time: {time:.4f}",
        "data: {data:.4f}"
    ])

    cpu_device = torch.device("cpu")
    model.eval()

    coco = coco_utils.get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = coco_eval.CocoEvaluator(coco, iou_types)

    start_time = time.time()
    end = time.time()
    i = 0
    for images, targets in data_loader:
        data_time = time.time() - end
        if log is not None:
            log.log("data_time", data_time)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        if log is not None:
            log.log("model_time", model_time)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        if log is not None:
            log.log("evaluator_time", evaluator_time)

        iter_time = time.time() - end
        if log is not None:
            log.log("iter_time", iter_time)
            if epoch is not None:
                log.commit(epoch=epoch, iteration=epoch * len(data_loader) + i)
            else:
                log.commit(iteration=i)
        if i % print_freq == 0 or i == len(data_loader) - 1:
            eta_seconds = iter_time * (len(data_loader) - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(log_msg.format(iter=i, total_iter=len(data_loader), eta=eta_string, model_time=model_time,
                                 eval_time=evaluator_time, time=iter_time, data=data_time))

        i += 1
        end = time.time()

    for iou_type in coco_evaluator.iou_types:
        coco_evaluator.eval_imgs[iou_type] = np.concatenate(coco_evaluator.eval_imgs[iou_type], 2)

        img_ids, idx = np.unique(coco_evaluator.img_ids, return_index=True)
        eval_imgs = coco_evaluator.eval_imgs[iou_type][..., idx]

        img_ids = list(coco_evaluator.img_ids)
        eval_imgs = list(coco_evaluator.eval_imgs[iou_type].flatten())
        coco_evaluator.coco_eval[iou_type].evalImgs = eval_imgs
        coco_evaluator.coco_eval[iou_type].params.imgIds = img_ids
        coco_evaluator.coco_eval[iou_type]._paramsEval = copy.deepcopy(coco_evaluator.coco_eval[iou_type].params)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    total_time = time.time() - start_time
    if log is not None:
        log.log("total_time", total_time)
        if epoch is not None:
            log.commit(epoch=epoch)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("{} Total time: {} ({:.4f}s / it)".format(head, total_time_str, total_time / len(data_loader)))
    return coco_evaluator


def main(args):
    dataset = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/train.json",
        get_transforms(True))
    dataset_val = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/val.json",
        get_transforms(False))
    dataset_test = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/test.json",
        get_transforms(False))
    dataset = torch.utils.data.Subset(dataset, [i for i in range(50)])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                             collate_fn=utils.collate_fn)
    dataset_val = torch.utils.data.Subset(dataset_val, [i for i in range(50)])
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)
    dataset_test = torch.utils.data.Subset(dataset_test, [i for i in range(50)])
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  collate_fn=utils.collate_fn)

    coco_api = coco_utils.get_coco_api_from_dataset(dataset)
    model = get_model_instance(num_classes=len(coco_api.cats) + 1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: {}".format(device.type))
    model.to(device)

    optimzier = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimzier, milestones=args.lr_steps, gamma=args.lr_gamma)
    epoch = 0

    log = Log()
    log_eval = Log(log_dir="log/eval")
    model_path_manager = ModelPathManager()

    latest_model_path = model_path_manager.latest_model_path()
    if latest_model_path:
        cpu_device = torch.device("cpu")
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
        optimzier.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch = checkpoint["epoch"] + 1
    elif args.pretrain_model:
        checkpoint = torch.load(args.pretrain_model)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.to(device)

    print("Start training")
    train_head = "Epoch : [{:0" + str(len(str(args.epochs))) + "d}]"
    start_time = time.time()
    for epoch in range(epoch, args.epochs):
        train_one_epoch(model, optimzier, data_loader, device, epoch, log, head=train_head.format(epoch))
        lr_scheduler.step()

        save_path = model_path_manager.new_model_path("train_epoch{:02d}.pth".format(epoch))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimzier.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }, save_path)
        model_path_manager.record_path(save_path)

        evaluate(model, data_loader_val, device, epoch, log_eval, head="Evaluate:")
        # engine.evaluate(model, data_loader_val, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    evaluate(model, data_loader_test, device, None, None, head="Test:")
    # engine.evaluate(model, data_loader_test, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lr-steps", default=[30, ], nargs='+', type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--pretrain-model")
    args = parser.parse_args()
    main(args)
