from dataset import TCTDataset
from pycocotools.cocoeval import COCOeval
import coco_utils

if __name__ == '__main__':
    image_set = "train"
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/{}.json".format(image_set))
    cocoGt = coco_utils.get_coco_api_from_dataset(dataset)
    cocoDt = cocoGt.loadRes("dataset/tct_result_{}.json".format(image_set))

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
