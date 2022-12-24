import os
import logging
import pickle
import torch
import torch.distributed as dist


class Log():
    def __init__(self):
        self.logger = None
        self.metric = None
        self.rank = dist.get_rank()

    def get_logger(self):
        self.logger = logging.getLogger(f"rank{self.rank}")
        self.logger.setLevel(logging.INFO)
        self.metric = dict()
        self.metric["rank"] = self.rank

    def add_handler(self, time):
        path = os.path.join("Log", time, f"rank{self.rank}")
        if os.path.exists(path) == False:
            os.makedirs(path)

        self.metric["path"] = path
        log_path = os.path.join(path, f"rank{self.rank}.log")
        fh = logging.FileHandler(log_path, mode='w')
        formatter = logging.Formatter("%(levelname)s - %(process)d - %(asctime)s - %(filename)s: %(message)s")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def add_metric(self, key, value):
        if key not in self.metric.keys():
            self.metric[key] = []
        self.metric[key].append(value)

    def save_metric(self):
        with open(os.path.join(self.metric["path"], f"rank{self.rank}.pkl"), "wb") as f:
            pickle.dump(self.metric, f)

    def save_model(self, model):
        model_path = os.path.join(self.metric["path"], "model_state_dict.pth")
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
