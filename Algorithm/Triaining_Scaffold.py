#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.generator import Generator
from models.Update import LocalUpdate_FedGen,DatasetSplit
from models.Fed import Aggregation
from models.test import test_img
from utils.utils import save_result
from optimizer.Adabelief import AdaBelief


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


class LocalUpdate_Scaffold(object):

    def __init__(self, args, state_params_diff=None, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.state_params_diff = state_params_diff
        self.max_norm = 10

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        if indd is not None:
            self.indd = indd
        else:
            self.indd = None

    def train(self,round, net, idx=-1):
        net.train()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
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
                for param in net.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * self.state_params_diff)

                loss = loss_fi + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(),
                                               max_norm=self.max_norm)  # Clip gradients to prevent exploding
                optimizer.step()

                num_updates += 1


        return net, num_updates

def Scaffold(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    weight_list = np.asarray([len(dict_users[i]) for i in range(args.num_users)])
    weight_list = weight_list / np.sum(weight_list) * args.num_users  # normalize it

    n_par = len(get_mdl_params([net_glob])[0])
    state_params_diffs = np.zeros((args.num_users + 1, n_par)).astype('float32')  # including cloud state

    # generate list of local models for each user
    w_locals = {}
    for user in range(args.num_users):

        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict


    acc = []
    lens = [len(datasets) for _,datasets in dict_users.items()]

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        delta_c_sum = np.zeros(n_par)
        prev_params = get_mdl_params([net_glob], n_par)[0]

        w_locals_selected = []
        selected_data_lens = []

        for idx in idxs_users:
            # Scale down c
            state_params_diff_curr = torch.tensor(
                -state_params_diffs[idx] + state_params_diffs[-1] / weight_list[idx], dtype=torch.float32,
                device=args.device)

            local = LocalUpdate_Scaffold(args=args, state_params_diff=state_params_diff_curr, dataset=dataset_train,
                                         idxs=dict_users[idx])

            net_local, count = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device), idx=idx)
            w_local = net_local.state_dict()
            w_locals_selected.append(w_local)
            selected_data_lens.append(lens[idx])

            curr_model_param = get_mdl_params([net_local], n_par)[0]
            new_c = state_params_diffs[idx] - state_params_diffs[-1] + 1 / count / args.lr * (
                        prev_params - curr_model_param)
            # Scale up delta c
            delta_c_sum += (new_c - state_params_diffs[idx]) * weight_list[idx]
            state_params_diffs[idx] = new_c

        # update global weights
        w_glob = Aggregation(w_locals_selected, selected_data_lens)
        state_params_diffs[-1] += 1 / args.num_users * delta_c_sum

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)

def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()