import copy
import random
from typing import List, OrderedDict, Dict, Any

import numpy as np
import torch
import wandb

from loguru import logger
from torch import nn

from Algorithm.Training import Training
from models import Aggregation, LocalUpdate_FedAvg, LocalUpdate_GitSFL
from models.SplitModel import Complete_ResNet18
from utils.utils import getTrueLabels

COMM_BUDGET = 0.1
DECAY = 0.5
DELTA = 0.01


@logger.catch
class GitSFL(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        # GitSFL Setting
        # self.net_glob = Complete_ResNet18(net_glob_client, net_glob_server)
        self.comm_budget = COMM_BUDGET
        self.repoSize = int(args.num_users * args.frac)
        self.repo = [copy.deepcopy(self.net_glob) for _ in range(self.repoSize)]
        self.modelServer = [copy.deepcopy(net_glob_server) for _ in range(self.repoSize)]
        self.modelClient = [copy.deepcopy(net_glob_client) for _ in range(self.repoSize)]
        self.cumulative_label_distributions = [np.zeros(args.num_classes) for _ in range(self.repoSize)]
        self.cumulative_label_distribution_weight = [0 for _ in range(self.repoSize)]
        self.true_labels = getTrueLabels(self)
        self.help_count = [0 for _ in range(args.num_users)]
        self.weakAggWeight = [1 for _ in range(self.repoSize)]

        self.grad_norm = [[0] * 12 for _ in range(self.repoSize)]
        self.Win = 10

        self.net_glob_client = net_glob_client
        self.net_glob_server = net_glob_server

        self.dataByLabel = self.organizeDataByLabel()

    @logger.catch()
    def train(self):
        while self.round < self.args.epochs:
            print("%" * 50)
            selected_users = np.random.choice(range(self.args.num_users), self.repoSize, replace=False)
            for modelIndex, client_index in enumerate(selected_users):
                self.cumulative_label_distribution_weight[modelIndex] = self.cumulative_label_distribution_weight[
                                                                            modelIndex] * DECAY + 1
                self.cumulative_label_distributions[modelIndex] = (self.cumulative_label_distributions[
                                                                       modelIndex] * DECAY +
                                                                   self.true_labels[client_index]) / \
                                                                  self.cumulative_label_distribution_weight[modelIndex]

                self.adjustBudget(modelIndex, client_index)

                self.splitTrain(client_index, modelIndex)
                # self.normalTrain(client_index, modelIndex)

            self.Agg()

            self.net_glob = Complete_ResNet18(self.net_glob_client, self.net_glob_server)
            self.test()

            for modelIndex in range(self.repoSize):
                self.weakAgg(modelIndex)

            self.round += 1

            if self.args.MR != 0:
                print(self.help_count)

    def splitTrain(self, curClient: int, modelIdx: int):
        sampledData = None
        if self.args.MR != 0:
            helpers, provide_data = self.selectHelpers(curClient, modelIdx)
            sampledData = self.sampleData(helpers, provide_data)

        local = LocalUpdate_GitSFL(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[curClient],
                                   helpers_idx=sampledData)
        mean_grad_norm = local.union_train(self.modelClient[modelIdx], self.modelServer[modelIdx])
        self.grad_norm[modelIdx].append(mean_grad_norm)

    def normalTrain(self, curClient: int, modelIdx: int):
        sampledData = self.dict_users[curClient]
        if self.args.MR == 1:
            helpers, provide_data = self.selectHelpers(curClient, modelIdx)
            sampledData = self.sampleData(helpers, provide_data)
            sampledData.extend(self.dict_users[curClient])
        local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=sampledData, verbose=False)
        w = local.train(round=self.round, net=copy.deepcopy(self.repo[modelIdx]).to(self.args.device))
        self.repo[modelIdx].load_state_dict(w)

    def Agg(self):
        w_client = [copy.deepcopy(model_client.state_dict()) for model_client in self.modelClient]
        w_avg_client = Aggregation(w_client, [1 for _ in range(self.repoSize)])
        self.net_glob_client.load_state_dict(w_avg_client)

        w_server = [copy.deepcopy(model_server.state_dict()) for model_server in self.modelServer]
        w_avg_server = Aggregation(w_server, [1 for _ in range(self.repoSize)])
        self.net_glob_server.load_state_dict(w_avg_server)

        #############################################
        # w = [copy.deepcopy(model.state_dict()) for model in self.repo]
        # w_avg = Aggregation(w, [1 for _ in range(self.repoSize)])
        # self.net_glob.load_state_dict(w_avg)

    def weakAgg(self, modelIdx: int):
        cur_model_client = self.modelClient[modelIdx]
        w = [copy.deepcopy(self.net_glob_client.state_dict()), copy.deepcopy(cur_model_client.state_dict())]
        lens = [self.weakAggWeight[modelIdx], 10]
        w_avg_client = Aggregation(w, lens)
        cur_model_client.load_state_dict(w_avg_client)

        cur_model_server = self.modelServer[modelIdx]
        w = [copy.deepcopy(self.net_glob_server.state_dict()), copy.deepcopy(cur_model_server.state_dict())]
        w_avg_server = Aggregation(w, lens)
        cur_model_server.load_state_dict(w_avg_server)

        ###########################################################
        # lens = [10, 1]
        # w = [copy.deepcopy(self.repo[modelIdx].state_dict()), copy.deepcopy(self.net_glob.state_dict())]
        # w_avg = Aggregation(w, lens)
        # self.repo[modelIdx].load_state_dict(w_avg)

    def sampleData(self, helper: int, provideData: List[int]) -> List[int]:
        # randomSample
        sampledData = []
        for classIdx, num in enumerate(provideData):
            sampledData.extend(random.sample(self.dataByLabel[helper][classIdx], num))
        return sampledData

    def selectHelpers(self, curClient: int, modelIdx: int) -> tuple[int, List[int]]:
        overall_requirement = max(10, int(len(self.dict_users[curClient]) * COMM_BUDGET))
        cumulative_label_distribution = self.cumulative_label_distributions[modelIdx]
        prior_of_classes = [max(np.mean(cumulative_label_distribution) - label, 0)
                            for label in cumulative_label_distribution]
        requirement_classes = [int(overall_requirement * (prior / sum(prior_of_classes))) for prior in prior_of_classes]

        helpers = 200
        provide_data = []
        max_contribution = 0
        candidate = list(range(self.args.num_users))
        candidate.pop(curClient)
        random.shuffle(candidate)
        weight = []
        data = []
        for client in candidate:
            contribution = 0
            temp = []
            for classIdx, label in enumerate(self.true_labels[client]):
                contribution += min(label, requirement_classes[classIdx])
                temp.append(min(label, requirement_classes[classIdx]))

            weight.append(contribution ** 2)
            data.append(temp)

            if contribution > max_contribution:
                max_contribution = contribution
                helpers = client
                provide_data = temp
        helpers = random.choices(candidate, weights=weight)[0]
        provide_data = data[candidate.index(helpers)]
        self.help_count[helpers] += 1

        print("-----MODEL #{}-----".format(modelIdx))
        print("overall_requirement:\t", overall_requirement)
        print("current_train_data:\t", list(self.true_labels[curClient]))
        print("cumu_label_distri:\t", list(map(int, cumulative_label_distribution)))
        print("prior_of_classes:\t", list(map(int, prior_of_classes)))
        print("required_classes:\t", requirement_classes)
        print("total_provide_data:\t", provide_data)
        print("selected_helper:\t", list(self.true_labels[helpers]))
        print("overall_supplement:\t", sum(provide_data))
        return helpers, provide_data

    def detectCLP(self, modelIdx) -> bool:
        OldNorm = max([np.mean(self.grad_norm[modelIdx][-self.Win - 1:-1]), 0.0000001])
        NewNorm = np.mean(self.grad_norm[modelIdx][-self.Win:])
        delta = (NewNorm - OldNorm) / OldNorm
        self.weakAggWeight[modelIdx] = 1 - delta
        # if modelIdx == 0 and OldNorm != 0:
        #     wandb.log({"round": self.round, "FGN": NewNorm, "delta": (NewNorm - OldNorm) / OldNorm})
        if delta > DELTA:
            return True
        return False

    def adjustBudget(self, modelIdx, clientIdx):
        CLP = self.detectCLP(modelIdx)
        global COMM_BUDGET
        if self.round < 100:
            COMM_BUDGET = 0.2
        elif self.round < 200:
            COMM_BUDGET = 0.15
        elif self.round < 500:
            COMM_BUDGET = 0.1
        else:
            COMM_BUDGET = 0
        pass

    def organizeDataByLabel(self) -> list[list[list[int]]]:
        organized = []
        for client in range(self.args.num_users):
            res = [[] for _ in range(self.args.num_classes)]
            all_local_data = self.dict_users[client]
            for data in all_local_data:
                res[self.dataset_train[data][1]].append(data)
            organized.append(res)
        return organized
