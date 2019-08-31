import os
import xml.dom.minidom
import json
import argparse

import metadata

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


info = {
    "year": "",
    "version": "",
    "description": "TCT",
    "contributor": "",
    "url": "",
    "date_created": ""
}

_license = {
    "id": 0,
    "name": "",
    "url": ""
}

_categories = metadata.categories
cat2id = metadata.cat2id

categories = [
    {
        "id": cat2id[name],
        "name": name,
        "supercategory": ""
    }
    for name in _categories
]

dataset = {
    "info": info,
    "images": [],
    "annotations": [],
    "licenses": [_license, ],
    "categories": categories
}


def load_ann(root, path):
    images = sorted(_load_images(os.path.join(root, path)))
    anns = []
    for image in images:
        xml_file = os.path.join(root, "Annotations", "{}.xml".format(image))
        annotation = _get_annotation(xml_file)
        anns.append(annotation)
    return anns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="the root of TCT data set")
    parser.add_argument("path", help="the path of image set list file relative to root")
    parser.add_argument("save_path", help="the path to save the annotations file")
    args = parser.parse_args()
    root = "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET"
    path = "ImageSets/Main/test.txt"
    image_id = 0
    ann_id = 0
    print("Loading annotations ...")
    for ann in load_ann(args.root, args.path):
        if not _is_valid_annotation(ann):
            print("{} is ignore".format(ann["filename"]))
            continue
        image_id += 1
        dataset["images"].append({
            "id": image_id,
            "width": ann["size"]["width"],
            "height": ann["size"]["height"],
            "file_name": "{}.jpg".format(ann["filename"]),
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        for obj in ann["object"]:
            ann_id += 1
            x1, y1, x2, y2 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
            dataset["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat2id[obj["name"]],
                "segmentation": [x1, y1, x1, y2, x2, y2, x2, y1],
                "area": (x2 - x1) * (y2 - y1),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0
            })

    with open(args.save_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print("Done")
