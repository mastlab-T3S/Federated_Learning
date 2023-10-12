import copy
import random
from typing import List

import numpy as np

from loguru import logger

from Algorithm.Training_ASync import Training_ASync
from models import Aggregation
from utils.utils import getTrueLabels

AMOUNT_OF_HELPERS = 3
AMOUNT_OF_ACTIVATION = 50


@logger.catch
class GitSFL(Training_ASync):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        # GitSFL Setting
        self.repoSize = int(args.num_users * args.frac)
        self.modelServer = []
        self.modelClient = []
        self.modelVersion = [0 for _ in range(self.repoSize)]
        self.true_labels = getTrueLabels(self)
        self.selected_count = [0 for _ in range(args.num_users)]

        self.net_glob_client = net_glob_client
        self.net_glob_server = net_glob_server

        self.distance = self.countDistance()
        self.dataByLabel = self.organizeDataByLabel()

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
            self.modelVersion[modelIndex] += 1
            for update in self.update_queue:
                update[-1] -= trainTime

            self.splitTrain(client_index)

            self.Agg()

            self.test()

            self.weakAgg(modelIndex)

            nextClient = self.selectNextClient()
            self.update_queue.append([nextClient, modelIndex, model_version + 1, self.clients.getTime(nextClient)])
            self.update_queue.sort(key=lambda x: x[-1])
            self.selected_count[nextClient] += 1
            self.idle_clients.remove(nextClient)
            self.idle_clients.add(client_index)

    def splitTrain(self, curClient: int):
        helpers = self.selectHelpers(curClient)
        sampledActivation = [self.sampleData(helper) for helper in helpers]

        pass

    def Agg(self):
        w_client = [copy.deepcopy(model_client.state_dict) for model_client in self.modelClient]
        w_avg_client = Aggregation(w_client, self.modelVersion)
        self.net_glob_client.load_state_dict(w_avg_client)

        w_server = [copy.deepcopy(model_server.state_dict) for model_server in self.modelClient]
        w_avg_server = Aggregation(w_server, self.modelVersion)
        self.net_glob_server.load_state_dict(w_avg_server)

        self.net_glob.load_state_dict(w_avg_client)
        self.net_glob.load_state_dict(w_avg_server)

    def selectNextClient(self) -> int:
        nextClient = random.choice(list(self.idle_clients))
        return nextClient

    def weakAgg(self, modelIdx: int):
        cur_model_client = self.modelClient[modelIdx]
        w = [copy.deepcopy(self.net_glob_client.state_dict()), copy.deepcopy(cur_model_client)]
        lens = [1, max(10 + self.modelVersion[modelIdx] - np.mean(self.modelVersion), 2)]
        w_avg_client = Aggregation(w, lens)
        cur_model_client.load_state_dict(w_avg_client)

        cur_model_server = self.modelServer[modelIdx]
        w = [copy.deepcopy(self.net_glob_server.state_dict()), copy.deepcopy(cur_model_server)]
        w_avg_server = Aggregation(w, lens)
        cur_model_client.load_state_dict(w_avg_server)

    def sampleData(self, helper: int) -> List[int]:
        # randomSample
        sampledNum = [int(AMOUNT_OF_ACTIVATION * (num / sum(self.true_labels[helper]))) for num in
                      self.true_labels[helper]]
        sampledData = []
        for classIdx, num in enumerate(sampledNum):
            sampledData.extend(random.sample(self.dataByLabel[helper][classIdx], num))
        return sampledData

    def selectHelpers(self, curClient: int) -> List[int]:
        curDistance = []
        for i, distance in enumerate(self.distance[curClient]):
            if i == curClient:
                continue
            curDistance.append((i, distance))

        curDistance.sort(key=lambda x: x[-1])
        helpers = [i[0] for i in curDistance[:AMOUNT_OF_HELPERS]]
        return helpers

    def countDistance(self) -> list[list[int]]:
        distance = [[-1 for _ in range(self.args.num_users)] for _ in range(self.args.num_users)]
        for i in range(self.args.num_users):
            for j in range(i + 1, self.args.num_users):
                dot = np.dot(self.true_labels[i], self.true_labels[j])
                distance[i][j] = distance[j][i] = dot
        return distance

    def organizeDataByLabel(self) -> list[list[list[int]]]:
        organized = []
        for client in self.args.num_users:
            res = [[] for _ in range(self.args.num_classes)]
            all_local_data = self.dict_users[client]
            for data in all_local_data:
                res[self.dataset_train[data][1]].append(data)
            organized.append(res)
        return organized
