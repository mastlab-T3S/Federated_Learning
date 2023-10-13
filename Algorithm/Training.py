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

        self.acc = []
        self.max_avg = 0
        self.max_std = 0

        if args.log:
            initWandb(args)

    def test(self):
        acc, loss = test(self.net_glob, self.dataset_test, self.args)
        self.acc.append(acc)
        temp = self.acc[max(0, len(self.acc) - 10)::]
        avg = np.mean(temp)
        if avg > self.max_avg:
            self.max_avg = avg
            self.max_std = np.std(temp)
        logger.info("Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}",
                    self.round, acc, self.max_avg, self.max_std, loss)
        if self.args.log:
            wandb.log({'acc': acc, 'max_avg': self.max_avg, "max_std": self.max_std, "loss": loss})


