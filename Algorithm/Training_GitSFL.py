import copy
import random
from typing import List

import numpy as np
import torch.nn
import wandb

from loguru import logger

from Algorithm.Training import Training
from models import Aggregation, LocalUpdate_GitSFL
from models.SplitModel import Complete_ResNet18
from utils.utils import getTrueLabels

BUDGET_THRESHOLD = 0.2
DECAY = 0.5
DELTA = 0
WIN = 10

COMM_BUDGET = 0.01
DATASET_SIZE = 50000
MODEL_SIZE = 614170
FEATURE_SIZE = int(13_107_622 / 50)


@logger.catch
class GitSFL(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        # GitSFL Setting
        self.traffic = 0
        self.comm_budget: float = COMM_BUDGET
        self.repoSize: int = int(args.num_users * args.frac)
        self.budget_list: List[float] = [COMM_BUDGET for _ in range(self.repoSize)]
        self.repo: List[torch.nn.Module] = [copy.deepcopy(self.net_glob) for _ in range(self.repoSize)]
        self.modelServer: List[torch.nn.Module] = [copy.deepcopy(net_glob_server) for _ in range(self.repoSize)]
        self.modelClient: List[torch.nn.Module] = [copy.deepcopy(net_glob_client) for _ in range(self.repoSize)]
        self.cumulative_label_distributions = [np.zeros(args.num_classes) for _ in range(self.repoSize)]
        self.cumulative_label_distribution_weight: List[float] = [0 for _ in range(self.repoSize)]
        self.true_labels = getTrueLabels(self)
        self.help_count = [0 for _ in range(args.num_users)]
        self.weakAggWeight = [1 for _ in range(self.repoSize)]

        self.grad_norm = [0 for _ in range(self.repoSize)]
        self.fed_grad_norm = [0 for _ in range(WIN + 2)]
        self.win = WIN
        self.helper_overhead = 0
        self.client_overhead = 0

        self.classify_count = [[[1] for _ in range(DATASET_SIZE)] for _ in range(self.repoSize)]

        self.net_glob_client = net_glob_client
        self.net_glob_server = net_glob_server

        self.dataByLabel = self.organizeDataByLabel()

    @logger.catch()
    def train(self):
        while (self.traffic / 1024 / 1024) < self.args.comm_limit:
            print("%" * 50)
            selected_users = np.random.choice(range(self.args.num_users), self.repoSize, replace=False)
            for modelIndex, client_index in enumerate(selected_users):
                self.cumulative_label_distribution_weight[modelIndex] = self.cumulative_label_distribution_weight[
                                                                            modelIndex] * DECAY + 1
                self.cumulative_label_distributions[modelIndex] = (self.cumulative_label_distributions[
                                                                       modelIndex] * DECAY +
                                                                   self.true_labels[client_index]) / \
                                                                  self.cumulative_label_distribution_weight[modelIndex]

                self.splitTrain(client_index, modelIndex)

            self.Agg()

            if self.args.DB:
                self.adjustBudget()

            self.net_glob = Complete_ResNet18(self.net_glob_client, self.net_glob_server)
            self.test()
            self.log()

            for modelIndex in range(self.repoSize):
                self.weakAgg(modelIndex)

            self.round += 1

            if self.args.MR != 0:
                print(self.help_count)

    def splitTrain(self, curClient: int, modelIdx: int):
        sampledData = None
        if self.args.MR != 0:
            helpers, provide_data = self.selectHelpers(curClient, modelIdx)
            sampledData = self.sampleData(helpers, provide_data, modelIdx)

        local = LocalUpdate_GitSFL(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[curClient],
                                   helpers_idx=sampledData)
        mean_grad_norm = local.union_train(self.modelClient[modelIdx], self.modelServer[modelIdx],
                                           self.classify_count[modelIdx])
        self.traffic += MODEL_SIZE * 2
        self.traffic += (len(self.dict_users[curClient]) * FEATURE_SIZE * self.args.local_ep * 2)
        self.grad_norm[modelIdx] = mean_grad_norm

    def Agg(self):
        w_client = [copy.deepcopy(model_client.state_dict()) for model_client in self.modelClient]
        w_avg_client = Aggregation(w_client, [1 for _ in range(self.repoSize)])
        self.net_glob_client.load_state_dict(w_avg_client)

        w_server = [copy.deepcopy(model_server.state_dict()) for model_server in self.modelServer]
        w_avg_server = Aggregation(w_server, [1 for _ in range(self.repoSize)])
        self.net_glob_server.load_state_dict(w_avg_server)

    def weakAgg(self, modelIdx: int):
        cur_model_client = self.modelClient[modelIdx]
        w = [copy.deepcopy(self.net_glob_client.state_dict()), copy.deepcopy(cur_model_client.state_dict())]
        lens = [1, 10]
        w_avg_client = Aggregation(w, lens)
        cur_model_client.load_state_dict(w_avg_client)

        cur_model_server = self.modelServer[modelIdx]
        w = [copy.deepcopy(self.net_glob_server.state_dict()), copy.deepcopy(cur_model_server.state_dict())]
        w_avg_server = Aggregation(w, lens)
        cur_model_server.load_state_dict(w_avg_server)

    def sampleData(self, helpers: List[int], provideData: List[List[int]], modexIdx: int) -> List[int]:
        # randomSample
        if self.args.BS == 0:
            sampledData = []
            for i, helper in enumerate(helpers):
                for classIdx, num in enumerate(provideData[i]):
                    sampledData.extend(random.sample(self.dataByLabel[helper][classIdx], num))
            return sampledData

        # boundarySample
        sampledData = []
        for i, helper in enumerate(helpers):
            for classIdx, num in enumerate(provideData[i]):
                if num == 0:
                    continue
                lst = [(dataIdx, np.mean(self.classify_count[modexIdx][dataIdx])) for dataIdx in
                       self.dataByLabel[helper][classIdx]]
                lst.sort(key=lambda x: x[-1])
                img = [i[0] for i in lst]
                w = [i[1] + 1e-10 for i in lst]
                w.reverse()
                sampledData.extend(random.choices(img, w, k=num))
        return sampledData

    def selectHelpers(self, curClient: int, modelIdx: int):
        overall_requirement = max(10, int(len(self.dict_users[curClient]) * COMM_BUDGET))
        cumulative_label_distribution = self.cumulative_label_distributions[modelIdx]
        prior_of_classes = [max(np.mean(cumulative_label_distribution) - label, 0)
                            for label in cumulative_label_distribution]
        requirement_classes = [int(overall_requirement * (prior / sum(prior_of_classes))) for prior in prior_of_classes]
        required = requirement_classes[::]

        helpers = []
        provide_data = []
        candidate = list(range(self.args.num_users))
        candidate.pop(curClient)
        random.shuffle(candidate)
        for client in candidate:
            if sum(requirement_classes) == 0:
                break
            temp = []
            for classIdx, label in enumerate(self.true_labels[client]):
                temp.append(min(label, requirement_classes[classIdx]))
                requirement_classes[classIdx] -= min(label, requirement_classes[classIdx])
            if sum(temp) > 0:
                self.help_count[client] += 1
                helpers.append(client)
                provide_data.append(temp)

        self.traffic += len(helpers) * MODEL_SIZE * self.args.local_ep
        self.traffic += (overall_requirement * FEATURE_SIZE * self.args.local_ep)
        self.helper_overhead += overall_requirement
        self.client_overhead += len(self.dict_users[curClient])

        print("-----MODEL #{}-----".format(modelIdx))
        print("overall_requirement:\t", overall_requirement)
        print("current_train_data:\t", list(self.true_labels[curClient]))
        print("cumu_label_distri:\t", list(map(int, cumulative_label_distribution)))
        print("prior_of_classes:\t", list(map(int, prior_of_classes)))
        print("required_classes:\t", required)
        print("total_provide_data:\t", provide_data)
        return helpers, provide_data

    def detectCLP(self):
        self.fed_grad_norm.append(np.mean(self.grad_norm))
        OldNorm = max([np.mean(self.fed_grad_norm[-self.win - 1:-1]), 0.0000001])
        NewNorm = np.mean(self.fed_grad_norm[-self.win:])
        delta = (NewNorm - OldNorm) / OldNorm
        return delta > DELTA, delta

        # self.weakAggWeight[modelIdx] = 1 - delta

    def adjustBudget(self):
        global COMM_BUDGET

        if self.args.DB == 1:
            CLP, delta = self.detectCLP()
            if CLP:
                if COMM_BUDGET >= BUDGET_THRESHOLD:
                    COMM_BUDGET += 0.01
                else:
                    COMM_BUDGET = min(BUDGET_THRESHOLD, COMM_BUDGET * 2)
                    # COMM_BUDGET = COMM_BUDGET * 2
            else:
                COMM_BUDGET = max(0.01, COMM_BUDGET / 2)
        elif self.args.DB == 2:
            CLP, delta = self.detectCLP()
            if self.round != 0:
                COMM_BUDGET = max(0.01, COMM_BUDGET * (1 + delta))

    def organizeDataByLabel(self) -> list[list[list[int]]]:
        organized = []
        for client in range(self.args.num_users):
            res = [[] for _ in range(self.args.num_classes)]
            all_local_data = self.dict_users[client]
            for data in all_local_data:
                res[self.dataset_train[data][1]].append(data)
            organized.append(res)
        return organized

    def log(self):
        logger.info(
            "Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}, comm:{:.2f}MB, budget:{:.2f}, add:{:.2f}",
            self.round, self.acc, self.max_avg, self.max_std,
            self.loss, (self.traffic / 1024 / 1024), COMM_BUDGET, (self.helper_overhead / self.client_overhead))
        if self.args.wandb:
            wandb.log({"round": self.round, 'acc': self.acc, 'max_avg': self.max_avg,
                       "max_std": self.max_std, "loss": self.loss,
                       "comm": (self.traffic / 1024 / 1024), "budget": COMM_BUDGET,
                       "add": (self.helper_overhead / self.client_overhead)})
