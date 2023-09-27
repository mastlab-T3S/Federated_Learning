#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
from random import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from models.Fed import *
from utils.utils import save_result
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=1.0, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

class LocalUpdate_FedNTD(object):
    def __init__(self, args, dg_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = NTD_Loss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.verbose = verbose
        self.dg_model = dg_model.eval()

    def train(self, net, round):

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
                model_output = net(images)
                dg_model_out = self._get_dg_logits(images)
                loss = self.loss_func(model_output['output'], labels, dg_model_out['output'])
                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)


        return net.state_dict()

    def _get_dg_logits(self, data):
        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits

def FedNTD(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedNTD(args=args, dg_model=copy.deepcopy(net_glob) ,dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device), round=iter)

            w_locals.append(copy.deepcopy(w))
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
