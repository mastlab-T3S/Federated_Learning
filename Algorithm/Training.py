import numpy as np
import wandb
from loguru import logger

from utils.utils import test, initWandb


class Training:

    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):

        self.round = 0
        self.args = args
        self.net_glob = net_glob
        self.net_glob.train()
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users

        self.acc_list = []
        self.acc = 0
        self.loss = 0
        self.max_avg = 0
        self.max_std = 0

        self.traffic = 0

        if args.wandb:
            initWandb(args)

    def test(self):
        self.acc, self.loss = test(self.net_glob, self.dataset_test, self.args)
        self.acc_list.append(self.acc)
        temp = self.acc_list[max(0, len(self.acc_list) - 10)::]
        avg = np.mean(temp)
        if avg > self.max_avg:
            self.max_avg = avg
            self.max_std = np.std(temp)

    def log(self):
        logger.info(
            "Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}, comm:{:.2f}MB",
            self.round, self.acc, self.max_avg, self.max_std,
            self.loss, (self.traffic / 1024 / 1024))
        if self.args.wandb:
            wandb.log({"round": self.round, 'acc': self.acc, 'max_avg': self.max_avg,
                       "max_std": self.max_std, "loss": self.loss,
                       "comm": (self.traffic / 1024 / 1024)})

