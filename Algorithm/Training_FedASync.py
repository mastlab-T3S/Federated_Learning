import copy

import numpy as np
from loguru import logger

from Algorithm.Training_ASync import Training_ASync
from models import LocalUpdate_FedProx, Aggregation


class FedASync(Training_ASync):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        m = max(int(args.frac * args.num_users), 1)
        init_users = self.clients.randomSampleClients(m)
        for model_index, client_index in enumerate(init_users):
            self.clients.train(copy.deepcopy(net_glob), self.round, client_index)

    def train(self):
        model_copy, version, update_idx, time = self.clients.getUpdate(1)[0]
        self.time += time

        local = LocalUpdate_FedProx(args=self.args, glob_model=model_copy, dataset=self.dataset_train,
                                    idxs=self.dict_users[update_idx])
        w = local.train(round=self.round, net=copy.deepcopy(self.net_glob).to(self.args.device))
        w_new = copy.deepcopy(self.net_glob.state_dict())

        lag = self.round - version
        alpha = self.args.FedASync_alpha * ((lag + 1) ** -self.args.poly_a)
        w_new = Aggregation([w, w_new], [alpha, 1 - alpha])
        self.net_glob.load_state_dict(w_new)

        self.test()

        self.log()

        nextClient = self.clients.randomSampleClients(1)[0]

        self.clients.train(copy.deepcopy(self.net_glob), self.round, nextClient)

    @logger.catch
    def run(self):
        while self.time < self.args.limit_time:
            self.train()
            self.round += 1
