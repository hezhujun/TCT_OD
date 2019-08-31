import json


class ResultRecorder(object):

    def __init__(self, save_path):
        self.save_path = save_path
        self.results = []

    def add(self, results):
        for image_id, v in results.items():
            boxes = v["boxes"].numpy()
            labels = v["labels"].numpy()
            scores = v["scores"].numpy()
            for box, label, score in zip(boxes, labels, scores):
                # print(box, label, score)
                res = {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "score": float(score),
                }
                self.results.append(res)

    def save(self, save_path=None):
        if save_path is not None:
            self.save_path = save_path
        with open(self.save_path, "w") as f:
            json.dump(self.results, f, indent=2)
