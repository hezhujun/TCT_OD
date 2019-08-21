import os
import xml.dom.minidom

import torch
from PIL import Image


def _load_images(path):
    files = []
    with open(path, "r") as f:
        _files = f.read().splitlines()
        for file in _files:
            if file.strip():
                files.append(file)
    return files


def _get_annotation(xml_file):
    DOMTree = xml.dom.minidom.parse(xml_file)
    root = DOMTree.documentElement
    annotation = {}
    folder = root.getElementsByTagName("folder")[0].childNodes[0].data
    annotation["folder"] = folder
    filename = root.getElementsByTagName("filename")[0].childNodes[0].data
    annotation["filename"] = filename
    sourceNode = root.getElementsByTagName("source")[0]
    source = {}
    database = sourceNode.getElementsByTagName("database")[0].childNodes[0].data
    _annoataion = sourceNode.getElementsByTagName("annotation")[0].childNodes[0].data
    image = sourceNode.getElementsByTagName("image")[0].childNodes[0].data
    flickrid = sourceNode.getElementsByTagName("flickrid")[0].childNodes[0].data
    source["database"] = database
    source["annotation"] = _annoataion
    source["image"] = image
    source["flickrid"] = flickrid
    annotation["source"] = source
    ownerNode = root.getElementsByTagName("owner")[0]
    owner = {}
    flickrid = ownerNode.getElementsByTagName("flickrid")[0].childNodes[0].data
    name = ownerNode.getElementsByTagName("name")[0].childNodes[0].data
    owner["flickrid"] = flickrid
    owner["name"] = name
    annotation["owner"] = owner
    sizeNode = root.getElementsByTagName("size")[0]
    size = {}
    width = sizeNode.getElementsByTagName("width")[0].childNodes[0].data
    height = sizeNode.getElementsByTagName("height")[0].childNodes[0].data
    depth = sizeNode.getElementsByTagName("depth")[0].childNodes[0].data
    size["width"] = int(width)
    size["height"] = int(height)
    size["depth"] = int(depth)
    annotation["size"] = size
    segmented = root.getElementsByTagName("segmented")[0].childNodes[0].data
    annotation["segmented"] = segmented
    objectNodes = root.getElementsByTagName("object")
    objects = []
    for objectNode in objectNodes:
        object = {}
        name = objectNode.getElementsByTagName("name")[0].childNodes[0].data
        object["name"] = name
        pose = objectNode.getElementsByTagName("pose")[0].childNodes[0].data
        object["pose"] = pose
        truncated = objectNode.getElementsByTagName("truncated")[0].childNodes[0].data
        object["truncated"] = truncated
        difficult = objectNode.getElementsByTagName("difficult")[0].childNodes[0].data
        object["difficult"] = difficult
        bndboxNode = objectNode.getElementsByTagName("bndbox")[0]
        bndbox = {}
        xmin = bndboxNode.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bndboxNode.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = bndboxNode.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bndboxNode.getElementsByTagName("ymax")[0].childNodes[0].data
        bndbox["xmin"] = float(xmin)
        bndbox["ymin"] = float(ymin)
        bndbox["xmax"] = float(xmax)
        bndbox["ymax"] = float(ymax)
        object["bndbox"] = bndbox
        objects.append(object)
    annotation["object"] = objects
    return annotation


def _get_boxes(annotation):
    boxes = []
    for object in annotation["object"]:
        bndbox = object["bndbox"]
        boxes.append([bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]])
    return boxes


def _is_valid_annotation(annotation):
    if len(annotation["object"]) == 0:
        return False
    for object in annotation["object"]:
        bndbox = object["bndbox"]
        if bndbox["xmin"] >= bndbox["xmax"] or bndbox["ymin"] >= bndbox["ymax"]:
            return False
    return True


class TCTDataset(object):

    def __init__(self, root, transform, label_transform, image_set="train"):
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        self.samples = []
        self.label2id = {}
        self.image_set = image_set
        if image_set == "train":
            images = sorted(_load_images(os.path.join(root, "ImageSets/Main/train.txt")))
        elif image_set == "val":
            images = sorted(_load_images(os.path.join(root, "ImageSets/Main/val.txt")))
        elif image_set == "test":
            images = sorted(_load_images(os.path.join(root, "ImageSets/Main/test.txt")))
        else:
            raise ValueError("image_set={} is not define".format(image_set))

        for image in images:
            xml_file = os.path.join(self.root, "Annotations", "{}.xml".format(image))
            annotation = _get_annotation(xml_file)
            if _is_valid_annotation(annotation):
                self.samples.append((image, annotation))

    def __getitem__(self, idx):
        image, annotation = self.samples[idx]
        image_file = os.path.join(self.root, "JPEGImages", "{}.jpg".format(image))
        image = Image.open(image_file).convert("RGB")

        target = {}
        num_objs = len(annotation["object"])
        boxes = torch.as_tensor(_get_boxes(annotation), dtype=torch.float32)
        labels = torch.as_tensor(self._get_labels(annotation), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor([idx, ])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.samples)

    def _get_labels(self, annotation):
        labels = []
        for object in annotation["object"]:
            label = object["name"]
            if self.label_transform is not None:
                label = self.label_transform(label)
            else:
                label_id = self.label2id.get(label, -1)
                if label_id == -1:
                    label_id = max([0, *self.label2id.values()]) + 1
                    self.label2id[label] = label_id
                label = label_id
            labels.append(label)
        return labels


# if __name__ == '__main__':
#     dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET", None, None)
#     for image, target in dataset:
#         print(target["area"])

