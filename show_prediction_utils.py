import cv2
from PIL import Image

import metadata
from tct_train import *

color_table = {
    0: (0x00, 0x00, 0x00),
    1: (0xFF, 0x00, 0x00),
    2: (0xFF, 0x99, 0x00),
    3: (0xFF, 0x00, 0xFF),
    4: (0x00, 0x00, 0xFF),
    5: (0x00, 0x66, 0x00),
    6: (0x66, 0x66, 0x00),
    7: (0x66, 0x33, 0x66),
    8: (0x00, 0x99, 0xCC),
    9: (0x99, 0x00, 0x99),
    10: (0x00, 0x33, 0x66),
    11: (0x33, 0x66, 0x33),
    12: (0x99, 0x99, 0x33),
    13: (0x99, 0x33, 0x99),
    14: (0xCC, 0xCC, 0x33),
    15: (0xD2, 0x69, 0x1E),
    16: (0x6A, 0x5A, 0xCD),
    17: (0x00, 0x80, 0x80),
    18: (0xD2, 0xB4, 0x8C),
}


def draw_dotted_rectangle(img, pt1, pt2, color, thickness, interval=10):
    width_points = list(np.arange(pt1[0], pt2[0], interval))
    if width_points[-1] != pt2[0]:
        width_points.append(pt2[0])

    height_points = list(np.arange(pt1[1], pt2[1], interval))
    if height_points[-1] != pt2[1]:
        height_points.append(pt2[1])

    for i in range(0, len(width_points), 2):
        if i + 1 >= len(width_points):
            break
        cv2.line(img, (width_points[i], pt1[1]), (width_points[i + 1], pt1[1]), color, thickness)
        cv2.line(img, (width_points[i], pt2[1]), (width_points[i + 1], pt2[1]), color, thickness)

    for i in range(0, len(height_points), 2):
        if i + 1 >= len(height_points):
            break
        cv2.line(img, (pt1[0], height_points[i]), (pt1[0], height_points[i + 1]), color, thickness)
        cv2.line(img, (pt2[0], height_points[i]), (pt2[0], height_points[i + 1]), color, thickness)


def draw_predictions(image, labels=None, prediction=None, category_ids=None, id2cat=None):
    if category_ids is None:
        category_ids = metadata.id2cat.keys()

    if id2cat is None:
        id2cat = metadata.id2cat

    # # the number of color bins
    # max_cat_id = max(category_ids)
    # num_bins = max_cat_id + 1
    # num_bins_per_channel = math.ceil(num_bins / 3)
    # act_num_bins_per_channel = 1
    # while num_bins_per_channel > act_num_bins_per_channel:
    #     act_num_bins_per_channel *= 2
    # color_step = (0xff + 1) // act_num_bins_per_channel

    if labels is not None:
        for label in labels:
            category_id = label["category_id"]
            color = color_table[category_id]
            bbox = label["bbox"]
            for i in range(len(bbox)):
                bbox[i] = round(bbox[i])
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, thickness=1)
            label_str = id2cat[category_id]
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_margin = 8
            text_height = text_size[1] + text_margin
            if bbox[1] - text_height < 0:
                cv2.putText(image, label_str, (bbox[0], bbox[1] + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(image, label_str, (bbox[0], bbox[1] - text_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if prediction is not None:
        for pred in prediction:
            category_id = pred["category_id"]
            color = color_table[category_id]
            score = pred["score"]
            bbox = pred["bbox"]
            for i in range(len(bbox)):
                bbox[i] = round(bbox[i])
            draw_dotted_rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, thickness=1)
            label_str = "{} {:.3f}".format(id2cat[category_id], score)
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_margin = 8
            text_height = text_size[1] + text_margin
            if bbox[1] - text_height < 0:
                cv2.putText(image, label_str, (bbox[0], bbox[1] + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(image, label_str, (bbox[0], bbox[1] - text_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


if __name__ == '__main__':
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT_DATASET/JPEGImages", "dataset/test.json")
    cocoGt = coco_utils.get_coco_api_from_dataset(dataset)
    cocoDt = cocoGt.loadRes("dataset/tct_result_test.json")

    # imgIds = sorted(cocoGt.getImgIds())
    # cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    # cocoEval.params.imgIds = imgIds
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    for idx in cocoGt.getImgIds():
        if idx < 80:
            continue
        ids = [idx]
        img = cocoGt.loadImgs(ids)[0]
        image = cv2.imread(os.path.join(dataset.root, img["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annIds = cocoGt.getAnnIds(imgIds=ids)
        anns = cocoGt.loadAnns(annIds)

        resIds = cocoDt.getAnnIds(imgIds=ids)
        res = cocoDt.loadAnns(resIds)

        image = draw_predictions(image, anns, res)
        image = Image.fromarray(image)
        image.show(title=img["file_name"])
        input()
