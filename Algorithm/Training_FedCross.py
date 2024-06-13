import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
from models.Fed import Aggregation
from utils.utils import save_result
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


class LocalUpdate_FedCross(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose

    def train(self, net):

        net.to(self.args.device)

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                predictive_loss = self.loss_func(model_output['output'], labels)

                loss = predictive_loss
                Predict_loss += predictive_loss.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        # net.to('cpu')

        return net.state_dict()


def FedCross(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []
    w_locals = []
    sim_arr = []

    m = max(int(args.frac * args.num_users), 1)
    for i in range(m):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):
            net_glob.load_state_dict(w_locals[i])
            local = LocalUpdate_FedCross(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_glob)
            w_locals[i] = copy.deepcopy(w)

        # update global weights
        w_glob = Aggregation(w_locals, None)  # Global Model Generation

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        sim_tab, sim_value = sim(args, w_locals)

        if iter % m == m - 1:
            sim_arr.append(sim_value)
            acc.append(test(net_glob, dataset_test, args))

        if iter >= args.fedcross_first_stage_bound:
            w_locals = cross_aggregation(args, iter, sim_tab, w_locals, m)  # Multi-Model Cross-Aggregation
        else:
            for i in range(len(w_locals)):
                w_locals[i] = copy.deepcopy(w_glob)

    save_result(acc, 'test_acc', args)
    save_result(sim_arr, 'sim', args)


def cross_aggregation(args, iter, sim_tab, w_locals, m):
    w_locals_new = copy.deepcopy(w_locals)

    crosslist = []

    for j in range(m):
        maxtag = 0
        submax = 1
        mintag = (j + 1) % m
        for p in range(m):
            if sim_tab[j][p] > sim_tab[j][maxtag]:
                submax = maxtag
                maxtag = p
            elif sim_tab[j][p] > sim_tab[j][submax]:
                submax = p

            if sim_tab[j][p] < sim_tab[j][mintag] and p != j:
                mintag = p

        rlist = []
        offset = iter % (m - 1) + 1
        sub_list = []
        alpha = args.fedcross_alpha
        select_strategy = args.fedcross_collaberative_model_select_strategy
        for k in range(m):
            if k == j:
                rlist.append(alpha)
                sub_list.append(copy.deepcopy(w_locals[j]))

            if select_strategy == 0:
                if (j + offset) % m == k:
                    rlist.append(1.0 - alpha)
                    sub_list.append(copy.deepcopy(w_locals[k]))
            elif select_strategy == 1:
                if mintag == k:
                    rlist.append(1.0 - alpha)
                    sub_list.append(copy.deepcopy(w_locals[mintag]))
            elif select_strategy == 2:
                if maxtag == k:
                    rlist.append(1.0 - alpha)
                    sub_list.append(copy.deepcopy(w_locals[maxtag]))
        w_cc = Aggregation(sub_list, rlist)
        crosslist.append(w_cc)

    for k in range(m):
        w_locals_new[k] = crosslist[k]

    return w_locals_new


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


def sim(args, net_glob_arr):
    model_num = int(args.num_users * args.frac)
    sim_tab = [[0 for _ in range(model_num)] for _ in range(model_num)]
    sum_sim = 0.0
    for k in range(model_num):
        sim_arr = []
        for j in range(k):
            s = 0.0
            dict_a = torch.Tensor(0)
            dict_b = torch.Tensor(0)
            cnt = 0
            for p in net_glob_arr[k].keys():
                a = net_glob_arr[k][p]
                b = net_glob_arr[j][p]
                a = a.view(-1)
                b = b.view(-1)

                if cnt == 0:
                    dict_a = a
                    dict_b = b
                else:
                    dict_a = torch.cat((dict_a, a), dim=0)
                    dict_b = torch.cat((dict_b, b), dim=0)

                if cnt % 2 == 0:
                    sub_a = a
                    sub_b = b
                else:
                    sub_a = torch.cat((sub_a, a), dim=0)
                    sub_b = torch.cat((sub_b, b), dim=0)

                if cnt % 2 == 1:
                    s += F.cosine_similarity(sub_a, sub_b, dim=0)
                cnt += 1
            s += F.cosine_similarity(sub_a, sub_b, dim=0)
            sim_arr.append(s)
            sim_tab[k][j] = s
            sim_tab[j][k] = s
            sum_sim += copy.deepcopy(s)
    l = int(len(net_glob_arr[0].keys()) / 5) + 1.0
    sum_sim /= (l * args.num_users * (args.num_users - 1) / 2.0)
    return sim_tab, sum_sim