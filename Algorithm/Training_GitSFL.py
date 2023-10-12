import copy

import numpy as np
import torch
from loguru import logger

from Algorithm.Training_ASync import Training_ASync


@logger.catch
class GitSFL(Training_ASync):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        # GitSFL Setting
        self.repoSize = int(args.num_users * args.frac)
        self.modelServer = []
        self.modelClient = []
        self.true_labels = self.getTrueLabels()
        self.selected_count = [0 for _ in range(args.num_users)]

    def getTrueLabels(self, dataset_train=None, num_classes=None, dict_users=None):
        trueLabels = []
        dataset_train = self.dataset_train if dataset_train is None else dataset_train
        num_classes = self.args.num_classes if num_classes is None else num_classes
        dict_users = self.dict_users if dict_users is None else dict_users
        for i in range(self.args.num_users):
            label = [0 for _ in range(num_classes)]
            for data_idx in dict_users[i]:
                label[dataset_train[data_idx][1]] += 1
            trueLabels.append(np.array(label))
        return trueLabels

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

    def test(self):
        pass

    def selectNextClient(self) -> int:
        pass

    def weakAgg(self):
        pass

    def sampleActivation(self) -> torch.tensor:
        pass
