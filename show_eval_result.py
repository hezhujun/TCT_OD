import matplotlib.pyplot as plt
import numpy as np

from _utils.log_utils import *

if __name__ == '__main__':
    entities = load_log_entities("log/eval/train-1567582236.dat")
    key_dict = extract_values(entities, "epoch", "metrics")
    metrics = key_dict["metrics"]
    epoch = key_dict["epoch"]

    metrics = np.array(metrics)
    AP1 = metrics[:, 0]
    AP2 = metrics[:, 1]
    AP3 = metrics[:, 2]

    plt.plot(epoch, AP1, label="AP IoU0.50:0.95")
    plt.plot(epoch, AP2, label="AP IoU0.50")
    plt.plot(epoch, AP3, label="AP IoU0.75")
    plt.legend()
    plt.show()
    print(np.argmax(AP2))
    print(metrics[-1])
