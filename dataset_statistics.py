import numpy as np
import matplotlib.pyplot as plt
from dataset import TCTDataset


def statistics(dataset):
    areas = []
    aspect_ratios = []
    for image, target in dataset:
        for ann in target:
            areas.append(ann["area"])
            aspect_ratios.append(ann["bbox"][3] / ann["bbox"][2])  # h / w

    plt.scatter(areas, aspect_ratios)
    plt.show()

    area_aspect = np.array([areas, aspect_ratios]).T
    return area_aspect


if __name__ == '__main__':
    dataset = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/train.json",
        None)
    dataset_val = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/val.json",
        None)
    dataset_test = TCTDataset(
        "/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/test.json",
        None)

    data = statistics(dataset)
    np.savetxt("dataset/area_aspect_train.txt", data)
    data = statistics(dataset_val)
    np.savetxt("dataset/area_aspect_val.txt", data)
    data = statistics(dataset_test)
    np.savetxt("dataset/area_aspect_test.txt", data)
