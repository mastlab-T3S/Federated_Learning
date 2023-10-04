import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from models import LocalUpdate_FedAvg, DatasetSplit
from optimizer.Adabelief import AdaBelief
from utils.utils import test
from models.Fed import Aggregation


class Demo:
    def __init__(self, args, dataset_train, dataset_test, proxy_dict, net_glob, dict_users):
        self.args = args
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.proxy_dict = proxy_dict
        self.net_glob = net_glob
        self.dict_users = dict_users

        self.M = max(int(self.args.frac * self.args.num_users), 1)
        self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]
        self.grad_glob = None

        self.round = 0

        self.acc = []
        self.max_avg = 0
        self.max_std = 0
        self.initWandb()

    def train(self):
        # 1. 选择设备
        # TODO 设备选择策略
        selected_users = np.random.choice(range(self.args.num_users), self.M, replace=False)

        # 2. 训练上传模型
        lens = []
        model_local = []  # 用于存储本地模型
        for trace_idx, client_idx in enumerate(selected_users):
            local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[client_idx])
            model = local.train(round=iter, net=copy.deepcopy(self.models[trace_idx]).to(self.args.device),
                                requestType="M")

            model_local.append(model)
            lens.append(len(self.dict_users[client_idx]))

        # 3. 本地模型互相蒸馏（9个其他模型加1个之前的模型）
        # TODO (1)蒸馏温度 (2)只用kl loss
        # model_local = self.mutualKD(model_local)

        # 4. 聚合蒸馏更新后的梯度
        grad = self.agg(model_local, lens)

        # 5. 增量相加梯度
        # self.grad_glob = grad
        self.accumulate(grad)

        # 5.1 更新全局模型
        for grad_idx, params in enumerate(self.net_glob.parameters()):
            params.data.add_(self.args.lr, self.grad_glob[grad_idx])

        # 6. TODO
        # (1) 直接用全局模型替换，不连续训练
        # self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]

        # (2) 蒸馏
        # self.models = model_local
        # self.kdWithNetGlob()

        # 2.1 蒸馏的时候只用klloss
        # self.models = model_local
        # self.kdWithNetGlob(0, 1)

        # (3) 弱聚合
        # weight = [10, 1]
        # self.models = model_local
        # for model in self.models:
        #     w = model.state_dict()
        #     w_avg = Aggregation([w, self.net_glob.state_dict()], weight)
        #     model.load_state_dict(w_avg)

        # (4) 设置连续训练次数
        if self.round != 0 and self.round % 10 == 0:
            self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]
        else:
            self.models = model_local

        # (5) 设置动态连续训练次数

        # (6) 什么都不做

    def mutualKD(self, models_local):
        afterKD = []
        for trace_index, model in enumerate(models_local):
            teachers = []
            for i in range(self.M):
                if i == trace_index:
                    continue
                teachers.append(models_local[i])
            teachers.append(self.models[trace_index])
            student = copy.deepcopy(model)
            self.KD(student, teachers)
            afterKD.append(student)
        return afterKD

    def agg(self, model_local, lens):
        grads = [self.getGrad(model_local[i], self.models[i]) for i in range(self.M)]
        agg_grad = None
        for i, grad in enumerate(grads):
            if i == 0:
                agg_grad = copy.deepcopy(grad)
                for j in range(len(agg_grad)):
                    agg_grad[j] = grad[j] * lens[i]
                continue
            for j in range(len(agg_grad)):
                agg_grad[j] += grad[j] * lens[i]

        total = sum(lens)
        for j in range(len(agg_grad)):
            agg_grad[j] = torch.div(agg_grad[j], total)
        return agg_grad

    def accumulate(self, grad):
        if self.grad_glob is None:
            self.grad_glob = grad
        else:
            for i in range(len(self.grad_glob)):
                self.grad_glob[i] = self.args.AC_alpha * self.grad_glob[i] + grad[i]

    def kdWithNetGlob(self):
        for model in self.models:
            self.KD(model, [self.net_glob])

    def klLoss(self, input_p, input_q, T=1):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        p = F.log_softmax(input_p / T, dim=1)
        q = F.softmax(input_q / T, dim=1)
        result = kl_loss(p, q)
        return result

    def KD(self, student, teachers):
        loss_func = nn.CrossEntropyLoss()
        ldr_train = DataLoader(DatasetSplit(self.dataset_train, self.proxy_dict), batch_size=self.args.local_bs,
                               shuffle=True, drop_last=True)
        student.train()
        # train and update
        optimizer = None
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(student.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(student.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(student.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for _ in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                student.zero_grad()
                input_p = student(images)['output']

                loss = 0
                if self.args.KD_alpha != 0:
                    loss = loss_func(input_p, labels)

                klLoss = 0
                for teacher in teachers:
                    input_q = teacher(images)['output']
                    klLoss += self.klLoss(input_p, input_q)
                klLoss /= len(teachers)

                loss = self.args.KD_beta * klLoss + loss * self.args.KD_alpha

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

    def getGrad(self, model, preModel):
        with torch.no_grad():
            delta = [para for para in model.parameters()]
            for grad_idx, params in enumerate(preModel.parameters()):
                delta[grad_idx] = (delta[grad_idx] - params) / self.args.lr
        return delta

    def test(self):
        acc = test(self.net_glob, self.dataset_test, self.args)
        self.acc.append(acc)
        temp = self.acc[max(0, len(self.acc) - 10)::]
        avg = np.mean(temp)
        if avg > self.max_avg:
            self.max_avg = avg
            self.max_std = np.std(temp)
        logger.info("Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}",
                    self.round, acc, self.max_avg, self.max_std)
        wandb.log({'acc': acc, 'max_avg': self.max_avg, "max_std": self.max_std})

    def initWandb(self):
        os.environ["WANDB_API_KEY"] = "ccea3a8394712aa6a0fd1eefd90832157836a985"
        data_split = "IID" if self.args.iid == 1 else str(self.args.data_beta)
        name = "{}_{}".format(data_split, self.args.algorithm)

        wandb.init(project="myFLWorkSpace", name=name,
                   tags=[str(self.args.model), str(self.args.dataset), data_split],
                   config={"seed": self.args.seed})
        wandb.log({'acc': 0, 'max_avg': 0, "max_std": 0})

    @logger.catch()
    def main(self):
        while self.round < self.args.epochs:
            self.train()
            self.test()
            self.round += 1
