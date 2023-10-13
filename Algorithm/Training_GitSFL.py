import copy
import math
import random
from typing import List, Mapping

import numpy as np

from loguru import logger

from Algorithm.Training_ASync import Training_ASync
from models import Aggregation
from utils.utils import getTrueLabels

AMOUNT_OF_HELPERS = 3
AMOUNT_OF_ACTIVATION = 50
COMM_BUDGET = 0.1


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
        self.clientsWeight = self.countClientsWeight()
        self.amount = [len(self.dict_users[clientIdx]) // weight for clientIdx, weight in enumerate(self.clientsWeight)]

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
            print("*" * 50)
            client_index, modelIndex, model_version, trainTime = self.update_queue.pop(0)
            self.modelVersion[modelIndex] += 1
            for update in self.update_queue:
                update[-1] -= trainTime

            print(self.true_labels[client_index])

            self.splitTrain(client_index)

            # self.Agg()

            # self.test()

            # self.weakAgg(modelIndex)

            nextClient = self.selectNextClient()
            self.update_queue.append([nextClient, modelIndex, model_version + 1, self.clients.getTime(nextClient)])
            self.update_queue.sort(key=lambda x: x[-1])
            self.selected_count[nextClient] += 1
            self.idle_clients.remove(nextClient)
            self.idle_clients.add(client_index)

    def splitTrain(self, curClient: int):
        helpers = self.selectHelpers(curClient)
        sampledData = [self.sampleData(helper, amount) for helper, amount in helpers]

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

    def sampleData(self, helper: int, amount: int) -> List[int]:
        # randomSample
        sampledNum = [int(amount * (num / sum(self.true_labels[helper]))) for num in
                      self.true_labels[helper]]
        print(sampledNum)
        sampledData = []
        for classIdx, num in enumerate(sampledNum):
            sampledData.extend(random.sample(self.dataByLabel[helper][classIdx], num))
        return sampledData

    def selectHelpers(self, curClient: int) -> Mapping[int, int]:
        # curDistance = []
        # for i, distance in enumerate(self.distance[curClient]):
        #     if i == curClient:
        #         continue
        #     curDistance.append((i, distance))
        #
        # curDistance.sort(key=lambda x: x[-1])
        # helpers = [i[0] for i in curDistance[:AMOUNT_OF_HELPERS]]

        budget = int(len(self.dict_users[curClient]) * COMM_BUDGET)

        amount = self.amount[::]
        amount.pop(curClient)

        weight = self.clientsWeight[::]
        weight.pop(curClient)

        value = self.distance[curClient][::]
        value.pop(curClient)
        value = [-v for v in value]

        print(self.args.num_users - 1, budget, amount, value, weight)
        max_value, selected_items, helpers = knapsack(self.args.num_users - 1, budget, amount, weight, value)
        print(max_value, selected_items)

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
        for client in range(self.args.num_users):
            res = [[] for _ in range(self.args.num_classes)]
            all_local_data = self.dict_users[client]
            for data in all_local_data:
                res[self.dataset_train[data][1]].append(data)
            organized.append(res)
        return organized

    def countClientsWeight(self) -> List[int]:
        clientsWeight = []
        for client in range(self.args.num_users):
            trueLabel = self.true_labels[client]
            union = 99999999
            print(trueLabel)
            for label in trueLabel:
                if label < union and label != 0:
                    union = label
            temp = [int(math.log(label // union)) for label in trueLabel]
            clientsWeight.append(sum(temp))
            print(temp)
            print(clientsWeight[-1])
        return clientsWeight


def knapsack(N, V, M, C, W):
    # 创建一个二维数组来保存动态规划的结果
    dp = [[0] * (V + 1) for _ in range(N + 1)]

    # 动态规划求解
    for i in range(1, N + 1):
        for j in range(1, V + 1):
            # 考虑第i种物品的情况
            for k in range(min(M[i - 1], j // C[i - 1]) + 1):
                dp[i][j] = max(dp[i][j], dp[i - 1][j - k * C[i - 1]] + k * W[i - 1])

    # 回溯找出装入背包的物品
    selected_items = {}
    helpers = {}
    i, j = N, V
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            cnt = min(M[i - 1], j // C[i - 1])
            selected_items[i] = [cnt, cnt * C[i - 1], cnt * W[i - 1]]
            helpers[i] = cnt * W[i - 1]
            j -= C[i - 1] * cnt
        i -= 1

    # 返回最大价值和选中的物品列表
    return dp[N][V], selected_items, helpers

# # 示例数据
# N = 4  # 物品种类数
# V = 5  # 背包容量
# M = [2, 1, 1, 2]  # 每种物品的最大件数
# C = [1, 2, 3, 2]  # 每件物品的空间消耗
# W = [3, 2, 4, 2]  # 每件物品的价值
#
# # 调用函数求解
# max_value, selected_items = knapsack(N, V, M, C, W)
# print(max_value)
# print(selected_items)
