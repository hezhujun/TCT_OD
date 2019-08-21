from torch.utils.data.dataloader import DataLoader
from dataset_voc import TCTDataset


if __name__ == '__main__':
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET", None, None)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    for images, targets in data_loader:
        break
