###############################################################################
# manage the paths of trained models
# generate the path of model
# record the path
# delete the useless model file
###############################################################################

import os


class ModelPathManager(object):

    def __init__(self, dir="log", max_file_path_size=5):
        self.dir = dir
        self.meta_data_file = ".checkpoint"
        self.file_paths = []
        assert max_file_path_size >=0, "max_file_path_size can not be negative"
        self.max_file_path_size = max_file_path_size

        self._init()

    def _init(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            return

        meta_data_file_path = os.path.join(self.dir, self.meta_data_file)
        if os.path.exists(meta_data_file_path):
            with open(meta_data_file_path, "r") as f:
                for path in f.read().splitlines():
                    if path.strip():
                        self.file_paths.append(path)

    def new_model_path(self, filename):
        return os.path.join(self.dir, filename)

    def record_path(self, path):
        self.file_paths.append(path)
        if self.max_file_path_size != 0 and len(self.file_paths) > self.max_file_path_size:
            path_to_delete = self.file_paths[:-self.max_file_path_size]
            for path in path_to_delete:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                self.file_paths.remove(path)
        self._update_meta_data()

    def latest_model_path(self):
        if len(self.file_paths) == 0:
            return None
        else:
            return self.file_paths[-1]

    def _update_meta_data(self):
        with open(os.path.join(self.dir, self.meta_data_file), "w") as f:
            for path in self.file_paths:
                f.write("{}\n".format(path))
