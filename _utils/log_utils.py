###############################################################################
# save log information by pickle
# the structure of pickle object is list with dict elements
###############################################################################

import time
import os
import pickle


class Log(object):

    def __init__(self, log_dir="log"):
        self.log_dir = log_dir
        self.datetime = int(time.time())
        self._filename = "train-{}.dat"
        self._log_entities = []
        self._current_entity = {}

    def log(self, key, value):
        self._current_entity[key] = value

    def commit(self, epoch=None, iteration=None):
        assert isinstance(epoch, (int, type(None)))
        assert isinstance(iteration, (int, type(None)))

        if epoch is None and iteration is None:
            raise ValueError("epoch and iteration are both None")

        if epoch is not None:
            self._current_entity["epoch"] = epoch
        if iteration is not None:
            self._current_entity["iteration"] = iteration

        self._log_entities.append(self._current_entity)
        self._current_entity = {}
        self._save()

    def _save(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        file_path = os.path.join(self.log_dir, self._filename.format(self.datetime))
        with open(file_path, "wb") as f:
            pickle.dump(self._log_entities, f)


def load_log_entities(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def extract_values(entities, *keys):
    key_dict = {}
    for key in keys:
        key_dict[key] = []
    key_set = set(keys)
    for entity in entities:
        if key_set.issubset(set(entity.keys())):
            for key in keys:
                key_dict[key].append(entity[key])

    return key_dict
