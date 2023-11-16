import numpy as np
import wandb
from loguru import logger

from Algorithm.Training import Training
from utils.ClientsHandler import ClientsHandler
from utils.utils import test


class Training_ASync(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        self.clients = ClientsHandler(args, dict_users)
        self.time = 0

    def log(self):
        logger.info(
            "Round{}, time:{:.2f}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}, comm:{:.2f}MB",
            self.round, self.time, self.acc, self.max_avg, self.max_std,
            self.loss, (self.traffic / 1024 / 1024))
        if self.args.wandb:
            wandb.log({"round": self.round, 'time': self.time, 'acc': self.acc, 'max_avg': self.max_avg,
                       "max_std": self.max_std, "loss": self.loss,
                       "comm": (self.traffic / 1024 / 1024)})
