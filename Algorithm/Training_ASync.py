import numpy as np
import wandb
from loguru import logger

from Algorithm.Training import Training
from utils.Clients import Clients
from utils.utils import test


class Training_ASync(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        self.update_queue = []
        self.clients = Clients(args, dict_users)
        self.train_time_list = self.clients.train_time
        self.idle_clients = set(list(range(args.num_users)))
        self.time = 0

    def test(self):
        acc, loss = test(self.net_glob, self.dataset_test, self.args)
        self.acc_list.append(acc)
        temp = self.acc_list[max(0, len(self.acc_list) - 10)::]
        avg = np.mean(temp)
        if avg > self.max_avg:
            self.max_avg = avg
            self.max_std = np.std(temp)
        logger.info("Round{}, time:{:.2f}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}",
                    self.round, self.time, acc, self.max_avg, self.max_std, loss)
        if self.args.log:
            wandb.log({'acc': acc, 'max_avg': self.max_avg, "max_std": self.max_std, "loss": loss, "time": self.time})
