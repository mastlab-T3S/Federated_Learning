from typing import List

import numpy as np
import torch
from loguru import logger

from Algorithm.Training_ASync import Training_ASync
from utils.utils import getTrueLabels


@logger.catch
class GitSFL(Training_ASync):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        # GitSFL Setting
        self.repoSize = int(args.num_users * args.frac)
        self.modelServer = []
        self.modelClient = []
        self.true_labels = getTrueLabels(self)
        self.selected_count = [0 for _ in range(args.num_users)]

    @logger.catch()
    def train(self):
        init_users = np.random.choice(range(self.args.num_users), self.repoSize, replace=False)
        for model_index, client_index in enumerate(init_users):
            # [client_index, modelIndex, model_version, trainTime]
            self.update_queue.append([client_index, model_index, 0, self.clients.getTime(client_index)])
            self.idle_clients.remove(client_index)
            self.selected_count[client_index] += 1
        self.update_queue.sort(key=lambda x: x[-1])

        while self.time < self.args.limit_time:
            client_index, modelIndex, model_version, trainTime = self.update_queue.pop(0)
            for update in self.update_queue:
                update[-1] -= trainTime

            self.splitTrain()

            self.Agg()

            self.test()

            self.weakAgg()

            nextClient = self.selectNextClient()
            self.update_queue.append([nextClient, modelIndex, model_version + 1, self.clients.getTime(nextClient)])
            self.update_queue.sort(key=lambda x: x[-1])
            self.selected_count[nextClient] += 1
            self.idle_clients.remove(nextClient)
            self.idle_clients.add(client_index)

    def splitTrain(self):
        sampledActivation = self.sampleActivation()
        pass

    def Agg(self):
        pass

    def selectNextClient(self) -> int:
        pass

    def weakAgg(self):
        pass

    def sampleActivation(self) -> List[torch.tensor]:
        pass
