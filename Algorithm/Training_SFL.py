import copy

import numpy as np
import wandb
from loguru import logger

from Algorithm.Training import Training
from models import LocalUpdate_SFL, Aggregation
from models.SplitModel import Complete_ResNet18

MODEL_SIZE = 614170
FEATURE_SIZE = int(13_107_622 / 50)


@logger.catch
class SFL(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        self.traffic = 0
        self.net_glob_client = net_glob_client
        self.net_glob_server = net_glob_server

    @logger.catch
    def train(self):
        while (self.traffic / 1024 / 1024) < self.args.comm_limit:
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            W_S = []
            W_C = []
            lens = []
            for idx in idxs_users:
                local = LocalUpdate_SFL(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                w_c, w_s = local.splitTrain(copy.deepcopy(self.net_glob_client), copy.deepcopy(self.net_glob_server))
                W_S.append(w_s)
                W_C.append(w_c)
                lens.append(len(self.dict_users[idx]))
            w_avg_client = Aggregation(W_C, lens)
            self.net_glob_client.load_state_dict(w_avg_client)
            w_avg_server = Aggregation(W_S, lens)
            self.net_glob_server.load_state_dict(w_avg_server)

            self.net_glob = Complete_ResNet18(self.net_glob_client, self.net_glob_server)
            self.test()

            self.traffic += MODEL_SIZE * 2 * m
            self.traffic += sum(lens) * FEATURE_SIZE * self.args.local_ep * 2

            self.log()

            self.round += 1
