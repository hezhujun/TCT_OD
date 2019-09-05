import matplotlib.pyplot as plt
import numpy as np

from _utils.log_utils import *

if __name__ == '__main__':
    entities = load_log_entities("log/train-1567390790.dat")
    key_dict = extract_values(entities, "iteration", "epoch", "loss", "loss_classifier", "loss_box_reg",
                              "loss_objectness", "loss_rpn_box_reg")
    iteration = np.array(key_dict["iteration"])
    epoch = np.array(key_dict["epoch"])
    loss = np.array(key_dict["loss"])
    loss_classifier = np.array(key_dict["loss_classifier"])
    loss_box_reg = np.array(key_dict["loss_box_reg"])
    loss_objectness = np.array(key_dict["loss_objectness"])
    loss_rpn_box_reg = np.array(key_dict["loss_rpn_box_reg"])

    epochs = epoch[-1] + 1

    epoch = np.split(epoch, epochs)
    epoch = np.array(epoch).mean(axis=1)
    loss = np.split(loss, epochs)
    loss = np.array(loss).mean(axis=1)
    loss_classifier = np.split(loss_classifier, epochs)
    loss_classifier = np.array(loss_classifier).mean(axis=1)
    loss_box_reg = np.split(loss_box_reg, epochs)
    loss_box_reg = np.array(loss_box_reg).mean(axis=1)
    loss_objectness = np.split(loss_objectness, epochs)
    loss_objectness = np.array(loss_objectness).mean(axis=1)
    loss_rpn_box_reg = np.split(loss_rpn_box_reg, epochs)
    loss_rpn_box_reg = np.array(loss_rpn_box_reg).mean(axis=1)

    plt.plot(epoch, loss, label="loss")
    plt.plot(epoch, loss_classifier, label="loss_classifier")
    plt.plot(epoch, loss_box_reg, label="loss_box_reg")
    plt.plot(epoch, loss_objectness, label="loss_objectness")
    plt.plot(epoch, loss_rpn_box_reg, label="loss_rpn_box_reg")
    plt.legend()
    plt.show()

    key_dict = extract_values(entities, "epoch", "total_time")
    plt.plot(key_dict["epoch"], key_dict["total_time"], label="total_time")
    plt.legend()
    plt.show()
