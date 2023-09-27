#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from models.Fed import Aggregation
from utils.utils import save_result
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


class LocalUpdate_FedDC(object):
    def __init__(self, args,  alpha, local_update_last, global_update_last, global_model_param, hist_i, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.alpha = alpha
        self.state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32,
                                         device=args.device)
        self.global_model_param = global_model_param
        self.hist_i = hist_i
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose
        self.max_norm = 10

    def train(self,round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Emsemble_loss = 0

        num_updates = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                loss_f_i = self.loss_func(model_output['output'], labels)

                local_parameter = None
                for param in net.parameters():
                    if not isinstance(local_parameter, torch.Tensor):
                        # Initially nothing to concatenate
                        local_parameter = param.reshape(-1)
                    else:
                        local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

                loss_cp = self.alpha / 2 * torch.sum((local_parameter - (self.global_model_param - self.hist_i)) * (
                            local_parameter - (self.global_model_param - self.hist_i)))
                loss_cg = torch.sum(local_parameter * self.state_update_diff)

                loss = loss_f_i + loss_cp + loss_cg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(),
                                               max_norm=self.max_norm)  # Clip gradients to prevent exploding
                optimizer.step()

                num_updates += 1

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)


        return net, num_updates


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


def set_client_from_params(mdl, params, device='cpu'):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def FedDC(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    weight_list = np.asarray([len(dict_users[i]) for i in range(args.num_users)])
    weight_list = weight_list / np.sum(weight_list) * args.num_users  # normalize it

    acc = []

    n_par = len(get_mdl_params([net_glob])[0])
    cld_mdl_param = get_mdl_params([net_glob], n_par)[0]

    ###
    parameter_drifts = np.zeros((args.num_users, n_par)).astype('float32')
    state_gadient_diffs = np.zeros((args.num_users + 1, n_par)).astype('float32')  # including cloud state

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=args.device)  # Theta
        delta_g_sum = np.zeros(n_par)

        for idx in idxs_users:

            local_update_last = state_gadient_diffs[idx]  # delta theta_i
            global_update_last = state_gadient_diffs[-1] / weight_list[idx]  # delta theta
            alpha = args.alpha_coef / weight_list[idx]
            hist_i = torch.tensor(parameter_drifts[idx], dtype=torch.float32, device=args.device)  # h_i

            local = LocalUpdate_FedDC(args=args, alpha=alpha, local_update_last=local_update_last, global_update_last=global_update_last,
                                            global_model_param=global_mdl, hist_i=hist_i, dataset=dataset_train, idxs=dict_users[idx])
            net_local, count = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            curr_model_par = get_mdl_params([net_local], n_par)[0]
            delta_param_curr = curr_model_par - cld_mdl_param
            parameter_drifts[idx] += delta_param_curr
            beta = 1 / count / args.lr

            state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
            delta_g_cur = (state_g - state_gadient_diffs[idx]) * weight_list[idx]
            delta_g_sum += delta_g_cur
            state_gadient_diffs[idx] = state_g

            w_locals.append(curr_model_par)
            lens.append(len(dict_users[idx]))

        # update global weights
        avg_mdl_param_sel = np.mean(np.array(w_locals), axis=0)
        delta_g_cur = 1 / args.num_users * delta_g_sum
        state_gadient_diffs[-1] += delta_g_cur

        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

        # copy weight to net_glob
        net_glob = set_client_from_params(net_glob, cld_mdl_param, args.device)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()
