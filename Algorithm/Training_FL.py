import copy

import numpy as np
import wandb
from loguru import logger

from Algorithm.Training import Training
from models import Aggregation, LocalUpdate_FedAvg

MODEL_SIZE = 44781080


class FL(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        self.traffic = 0

    @logger.catch
    def train(self):
        while (self.traffic / 1024 / 1024) < self.args.comm_limit:
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            lens = []
            w = []
            for idx in idxs_users:
                local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                w_local = local.train(round=None, net=copy.deepcopy(self.net_glob))
                lens.append(len(self.dict_users[idx]))
                w.append(w_local)

            self.net_glob.load_state_dict(Aggregation(w, lens))
            self.test()

            self.traffic += MODEL_SIZE * 2 * m

            self.log()

            self.round += 1
