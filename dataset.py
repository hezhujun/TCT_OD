import torchvision


class TCTDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        super(TCTDataset, self).__init__(root, ann_file, transforms=transforms)

    def __getitem__(self, idx):
        img, target = super(TCTDataset, self).__getitem__(idx)
        return img, target


if __name__ == '__main__':
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/val.json")
    for img, target in dataset:
        print(target)
        exit()
