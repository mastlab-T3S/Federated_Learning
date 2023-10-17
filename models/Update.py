#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import math

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from optimizer.Adabelief import AdaBelief


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # if len(idxs) % args.local_bs != 1:
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # else:
        #     self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
        #                                 drop_last=True)
        self.verbose = verbose

    def train(self, round, net, requestType="W"):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            if self.args.dynamic_lr == 1:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            else:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if len(labels) == 1:
                    pass
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)
        if requestType == "W":
            return net.state_dict()
        else:
            return net


class LocalUpdate_ClientSampling(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net


class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Penalize_loss = 0

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += (
                            (self.prox_alpha / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss = predictive_loss + fed_prox_reg
                Predict_loss += predictive_loss.item()
                Penalize_loss += fed_prox_reg.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Penalize loss={:.4f}'.format(Penalize_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_Scaffold(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if indd is not None:
            self.indd = indd
        else:
            self.indd = None

    def train(self, round, net, c_list={}, idx=-1):
        net.train()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        num_updates = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = net(images)['output']
                loss_fi = self.loss_func(log_probs, labels)

                local_par_list = None
                dif = None
                for param in net.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                for k in c_list[idx].keys():
                    if not isinstance(dif, torch.Tensor):
                        dif = (-c_list[idx][k] + c_list[-1][k]).reshape(-1)
                    else:
                        dif = torch.cat((dif, (-c_list[idx][k] + c_list[-1][k]).reshape(-1)), 0)
                loss_algo = torch.sum(local_par_list * dif)
                loss = loss_fi + loss_algo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()

                num_updates += 1

        return net.state_dict(), num_updates


class LocalUpdate_FedGKD(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.glob_model = glob_model.to(args.device)
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Emsemble_loss = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                global_output_logp = self.glob_model(images)['output']

                user_latent_loss = self.ensemble_alpha * self.ensemble_loss(F.log_softmax(log_probs, dim=1),
                                                                            F.softmax(global_output_logp, dim=1))

                loss = predictive_loss + user_latent_loss
                Predict_loss += predictive_loss.item()
                Emsemble_loss += user_latent_loss.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Emsemble loss={:.4f}'.format(Emsemble_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_Moon(object):
    def __init__(self, args, glob_model, old_models, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.glob_model = glob_model.to(args.device)
        self.old_models = old_models
        self.contrastive_alpha = args.contrastive_alpha
        self.temperature = args.temperature
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Contrastive_loss = 0

        for iter in range(self.args.local_ep):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                output = net(images)
                predictive_loss = self.loss_func(output['output'], labels)

                output_representation = output['representation']
                pos_representation = self.glob_model(images)['representation']
                posi = self.cos(output_representation, pos_representation)
                logits = posi.reshape(-1, 1)

                for previous_net in self.old_models:
                    neg_representation = previous_net(images)['representation']
                    nega = self.cos(output_representation, neg_representation)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.temperature
                labels = torch.zeros(images.size(0)).to(self.args.device).long()

                contrastive_loss = self.contrastive_alpha * self.loss_func(logits, labels)

                loss = predictive_loss + contrastive_loss
                Predict_loss += predictive_loss.item()
                Contrastive_loss += contrastive_loss.item()

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(predictive_loss.item())
                epoch_loss2_collector.append(contrastive_loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            if self.verbose:
                print('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (iter, epoch_loss, epoch_loss1, epoch_loss2))

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Contrastive loss={:.4f}'.format(Contrastive_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_FedGen(object):
    def __init__(self, args, generative_model, dataset=None, idxs=None, verbose=False, regularization=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction='batchmean')
        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose
        self.generative_model = generative_model
        self.regularization = regularization
        self.generative_alpha = args.generative_alpha
        self.generative_beta = args.generative_beta
        self.latent_layer_idx = -1

    def train(self, round, net):

        net.train()
        self.generative_model.eval()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Teacher_loss = 0
        Latent_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, y) in enumerate(self.ldr_train):
                images, y = images.to(self.args.device), y.to(self.args.device)
                net.zero_grad()
                user_output_logp = net(images)['output']
                predictive_loss = self.loss_func(user_output_logp, y)

                #### sample y and generate z
                if self.regularization:
                    ### get generator output(latent representation) of the same label
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output'].to(
                        self.args.device)
                    logit_given_gen = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = self.generative_beta * self.ensemble_loss(F.log_softmax(user_output_logp, dim=1),
                                                                                 target_p)

                    sampled_y = np.random.choice(self.args.num_classes, self.args.bs)
                    sampled_y = torch.LongTensor(sampled_y).to(self.args.device)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output'].to(
                        self.args.device)  # latent representation when latent = True, x otherwise
                    user_output_logp = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = self.generative_alpha * torch.mean(
                        self.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.args.bs / self.args.bs
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    Teacher_loss += teacher_loss.item()
                    Latent_loss += user_latent_loss.item()
                else:
                    #### get loss and perform optimization
                    loss = predictive_loss

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = 'User predict Loss={:.4f} Teacher Loss={:.4f} Latent Loss={:.4f}'.format(
                Predict_loss / (self.args.local_ep * len(self.ldr_train)),
                Teacher_loss / (self.args.local_ep * len(self.ldr_train)),
                Latent_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        net.to('cpu')

        return net


class LocalUpdate_GitSFL:
    # 初始化组，参数依次为组内的客户端ID列表，学习率，设备（GPU)，完整的训练集，组内客户端数据索引，batch个数，分组策略
    def __init__(self, args, dataset=None, idxs=None, helpers_idx=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_helper = []
        if helpers_idx is not None:
            self.ldr_train_helper = DataLoader(DatasetSplit(dataset, helpers_idx),
                                               batch_size=max(math.ceil(len(helpers_idx) / len(self.ldr_train)), 2),
                                               shuffle=True,
                                               drop_last=True)
        self.loss_func = nn.CrossEntropyLoss()

    def union_train(self, net_client, net_server):
        net_client.train()
        net_server.train()
        # train and update
        optimizer_client = torch.optim.SGD(net_client.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_server = torch.optim.SGD(net_server.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        GNorm = []
        for iter in range(self.args.local_ep):
            grad_norm = 0
            # 由于每个客户端的batch_len一致，遍历每一个batch
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # 保存所有数据计算出中间特征
                all_fx = []
                # 保存所有数据的label
                all_labels = torch.tensor([]).to(self.args.device)

                # 计算client的特征
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                fx_client = net_client(images)
                all_fx.append(fx_client)
                all_labels = torch.cat([all_labels, labels], axis=0)

                # 计算helper的特征
                images_helper, labels_helper, fx_helper = None, None, None
                if batch_idx < len(self.ldr_train_helper):
                    for count, (i, l) in enumerate(self.ldr_train_helper):
                        images_helper, labels_helper = i, l
                        if count == batch_idx:
                            break
                    images_helper, labels_helper = images_helper.to(self.args.device), labels_helper.to(
                        self.args.device)
                if images_helper is not None:
                    temp_net = copy.deepcopy(net_client)
                    fx_helper = temp_net(images_helper)
                    all_fx.append(fx_helper)
                    all_labels = torch.cat([all_labels, labels_helper], axis=0)

                net_client.zero_grad()

                fx_to_server = [fx.clone().detach().requires_grad_(True) for fx in all_fx]
                all_labels = all_labels.to(torch.int64)

                net_server.zero_grad()
                fx_server = torch.tensor([]).to(self.args.device)
                for i, fx in enumerate(fx_to_server):
                    part_fx_server = net_server(fx)
                    fx_server = torch.cat([fx_server, part_fx_server], axis=0)
                loss = self.loss_func(fx_server, all_labels)
                loss.backward()
                optimizer_server.step()

                all_dfx = [fx.grad.clone().detach() for fx in fx_to_server]
                for i, fx in enumerate(all_fx):
                    fx.backward(all_dfx[i])
                optimizer_client.step()

                temp_norm = 0
                for parms in net_client.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                grad_norm = grad_norm + temp_norm

                for parms in net_server.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                grad_norm = grad_norm + temp_norm
            GNorm.append(grad_norm)
        return np.mean(GNorm) * self.args.lr
