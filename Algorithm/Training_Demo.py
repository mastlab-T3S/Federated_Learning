# import copy
# import os
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# import wandb
# from loguru import logger
# from torch import nn
# from torch.utils.data import DataLoader
#
# from models import LocalUpdate_FedAvg, DatasetSplit
# from optimizer.Adabelief import AdaBelief
# from utils.get_dataset import get_dataset
# from utils.utils import test, initWandb
# from models.Fed import Aggregation
# from tqdm import trange
#
# KD_epoch = 1
#
#
# class Demo:
#     def __init__(self, args, dataset_train, dataset_test, proxy_dict, net_glob, dict_users):
#         self.args = args
#         self.dataset_train = dataset_train
#         self.dataset_test = dataset_test
#         self.proxy_dict = proxy_dict
#         self.net_glob = net_glob
#         self.dict_users = dict_users
#
#         self.M = max(int(self.args.frac * self.args.num_users), 1)
#         self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]
#         self.grad_glob = None
#         self.true_labels = self.getTrueLabels()
#
#         self.round = 0
#
#         self.acc = []
#         self.max_avg = 0
#         self.max_std = 0
#         initWandb(args)
#
#         args.iid = 1
#         # args.data_beta = 0.1
#         new_dataset_train, new_dataset_test, new_dict_users = get_dataset(args)
#         self.new_dict_users = new_dict_users
#
#     def getTrueLabels(self, normal=False, dataset_train=None, num_classes=None, dict_users=None):
#         trueLabels = []
#         dataset_train = self.dataset_train if dataset_train is None else dataset_train
#         num_classes = self.args.num_classes if num_classes is None else num_classes
#         dict_users = self.dict_users if dict_users is None else dict_users
#         for i in range(self.args.num_users):
#             label = [0 for _ in range(num_classes)]
#             for data_idx in dict_users[i]:
#                 label[dataset_train[data_idx][1]] += 1
#             # if normal:
#             #     label = unitization(np.array(label))
#             trueLabels.append(np.array(label))
#         return trueLabels
#
#     def train(self):
#         self.net_glob.train()
#         # 1. 选择设备
#         # TODO 设备选择策略
#         selected_users = list(np.random.choice(range(self.args.num_users), self.M, replace=False))
#         if self.round > 200:
#             self.dict_users = self.new_dict_users
#         # 2. 训练上传模型
#         lens = []
#         model_local = []  # 用于存储本地模型
#         for trace_idx, client_idx in enumerate(selected_users):
#             local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[client_idx],
#                                        verbose=False)
#             model = local.train(round=self.round, net=copy.deepcopy(self.models[trace_idx]).to(self.args.device),
#                                 requestType="M")
#
#             model_local.append(copy.deepcopy(model))
#             lens.append(len(self.dict_users[client_idx]))
#
#         # 3. 本地模型互相蒸馏（9个其他模型加1个之前的模型）
#         # TODO (1)蒸馏温度 (2)只用kl loss
#         # model_local = self.mutualKD(model_local, selected_users)
#
#         # 4. 聚合蒸馏更新后的梯度
#         grad = self.agg(model_local, lens)
#
#         # 5. 增量相加梯度
#         self.grad_glob = grad
#         # self.accumulate(grad)
#
#         # 5.1 更新全局模型
#         w_glob = copy.deepcopy(self.net_glob.state_dict())
#         for k in w_glob.keys():
#             if w_glob[k].dtype != torch.long:
#                 w_glob[k] += self.args.lr * self.grad_glob[k]
#             else:
#                 w_glob[k] = w_glob[k].float() + self.args.lr * self.grad_glob[k]
#         self.net_glob.load_state_dict(w_glob)
#
#         # 6. TODO
#         # (1) 直接用全局模型替换，不连续训练
#         # self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]
#
#         # (2) 蒸馏
#         # self.models = model_local
#         # self.kdWithNetGlob()
#
#         # 2.1 蒸馏的时候只用klloss
#         # self.models = model_local
#         # self.kdWithNetGlob(0, 1)
#
#         # (3) 弱聚合
#         # weight = [10, 1]
#         # self.models = model_local
#         # for model in self.models:
#         #     w = model.state_dict()
#         #     w_avg = Aggregation([w, self.net_glob.state_dict()], weight)
#         #     model.load_state_dict(w_avg)
#
#         # (4) 设置连续训练次数
#         if self.round != 0 and self.round % 10 == 0:
#             self.models = [copy.deepcopy(self.net_glob) for _ in range(self.M)]
#         # elif self.round < 100:
#         #     # 弱聚合
#         #     for model in model_local:
#         #         w = [model.state_dict(), self.net_glob.state_dict()]
#         #         lens = [10, 1]
#         #         w_avg = Aggregation(w, lens)
#         #         model.load_state_dict(w_avg)
#         #     self.models = model_local
#         else:
#             self.models = model_local
#
#         # (5) 设置动态连续训练次数
#
#         # (6) 什么都不做
#
#     def mutualKD(self, models_local, selected_users):
#         afterKD = []
#
#         # for trace_index, model in enumerate(models_local):
#         #     client_idx = selected_users[trace_index]
#         #     lst = []
#         #     for i, client in enumerate(selected_users):
#         #         if client != client_idx:
#         #             lst.append((i, np.dot(self.true_labels[client_idx], self.true_labels[client])))
#         #     lst.sort(key=lambda x: x[1])
#         #     teachers = [models_local[lst[0][0]], self.models[trace_index]]
#         #     student = copy.deepcopy(model)
#         #     self.KD(student, teachers)
#         #     afterKD.append(student)
#         # return afterKD
#
#         for trace_index, model in enumerate(models_local):
#             teachers = []
#             for i in range(self.M):
#                 if i == trace_index:
#                     continue
#                 teachers.append(models_local[i])
#             teachers.append(self.models[trace_index])
#             student = copy.deepcopy(model)
#             self.KD(student, teachers)
#             afterKD.append(student)
#         return afterKD
#
#     def agg(self, model_local, lens):
#         grads = [self.getGrad(model_local[i], self.models[i]) for i in range(self.M)]
#         return Aggregation(grads, lens)
#
#     def accumulate(self, grad):
#         if self.grad_glob is None:
#             self.grad_glob = grad
#         else:
#             for k in self.grad_glob.keys():
#                 self.grad_glob[k] = self.args.AC_alpha * self.grad_glob[k] + grad[k]
#
#     def kdWithNetGlob(self):
#         for model in self.models:
#             self.KD(model, [self.net_glob])
#
#     def klLoss(self, input_p, input_q, T=1):
#         kl_loss = nn.KLDivLoss(reduction="batchmean")
#         p = F.log_softmax(input_p / T, dim=1)
#         q = F.softmax(input_q / T, dim=1)
#         result = kl_loss(p, q)
#         return result
#
#     def KD(self, student, teachers):
#         loss_func = nn.CrossEntropyLoss()
#         ldr_train = DataLoader(DatasetSplit(self.dataset_train, self.proxy_dict), batch_size=self.args.local_bs,
#                                shuffle=True, drop_last=True)
#         student.train()
#         # train and update
#         optimizer = None
#         if self.args.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(student.parameters(), lr=self.args.KD_lr, momentum=self.args.momentum)
#         elif self.args.optimizer == 'adam':
#             optimizer = torch.optim.Adam(student.parameters(), lr=self.args.lr)
#         elif self.args.optimizer == 'adaBelief':
#             optimizer = AdaBelief(student.parameters(), lr=self.args.lr)
#
#         Predict_loss = 0
#         for _ in range(self.args.KD_epoch):
#             for batch_idx, (images, labels) in enumerate(ldr_train):
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 student.zero_grad()
#                 input_p = student(images)['output']
#
#                 loss = 0
#                 if self.args.KD_alpha != 0:
#                     loss = loss_func(input_p, labels)
#
#                 klLoss = 0
#                 for teacher in teachers:
#                     input_q = teacher(images)['output']
#                     klLoss += self.klLoss(input_p, input_q)
#                 klLoss /= len(teachers)
#
#                 if self.args.KD_alpha != 0:
#                     loss = self.args.KD_beta * klLoss + loss * self.args.KD_alpha
#                 else:
#                     loss = klLoss
#
#                 loss.backward()
#                 optimizer.step()
#
#                 Predict_loss += loss.item()
#
#     def getGrad(self, model, preModel):
#         with torch.no_grad():
#             delta = copy.deepcopy(model.state_dict())
#             w = preModel.state_dict()
#             for k in w.keys():
#                 delta[k] = (delta[k] - w[k]) / self.args.lr
#         return delta
#
#     def test(self):
#         acc, loss = test(self.net_glob, self.dataset_test, self.args)
#         self.acc.append(acc)
#         temp = self.acc[max(0, len(self.acc) - 10)::]
#         avg = np.mean(temp)
#         if avg > self.max_avg:
#             self.max_avg = avg
#             self.max_std = np.std(temp)
#         logger.info("Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}",
#                     self.round, acc, self.max_avg, self.max_std, loss)
#         wandb.log({'acc': acc, 'max_avg': self.max_avg, "max_std": self.max_std, "loss": loss})
#
#     @logger.catch()
#     def main(self):
#         while self.round < self.args.epochs:
#             self.train()
#             self.test()
#             self.round += 1
