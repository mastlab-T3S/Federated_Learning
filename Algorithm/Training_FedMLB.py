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

def KD(input_p, input_q, T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p/T, dim = 1)
    q = F.log_softmax(input_q/T, dim = 1)
    result = kl_loss(q,p)
    return result

class LocalUpdate_FedMLB(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.glob_model = glob_model.to(args.device)
        self.contrastive_alpha = args.contrastive_alpha
        self.temperature = args.temperature
        self.verbose = verbose

    def train(self,round, net):

        net.train()
        self.glob_model.eval()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                out_of_local = net(images)
                log_probs = out_of_local['output']
                log_prob_branch = []
                ce_branch = []
                kl_branch = []
                num_branch = 5

                ## Compute loss from hybrid branches
                for it in range(num_branch):
                    if it!=4:
                        this_log_prob = self.glob_model(out_of_local['representation' + str(it)], start_layer_idx = it + 1)
                    else:
                        this_log_prob = self.glob_model(out_of_local['representation'], start_layer_idx=it + 1)
                    this_ce = self.loss_func(this_log_prob, labels)
                    this_kl = KD(this_log_prob, log_probs, self.args.temp)
                    log_prob_branch.append(this_log_prob)
                    ce_branch.append(this_ce)
                    kl_branch.append(this_kl)

                ce_loss = self.loss_func(log_probs, labels)
                loss = self.args.lambda1 * ce_loss + self.args.lambda2 * (
                    sum(ce_branch)) / num_branch + self.args.lambda3 * (sum(kl_branch)) / num_branch
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict()


def FedMLB(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []


    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        for idx in idxs_users:
            local = LocalUpdate_FedMLB(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)

            w_local = local.train(round=iter, net=net_local.to(args.device))

            w_locals.append(w_local)
            lens.append(len(dict_users[idx]))

        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()
