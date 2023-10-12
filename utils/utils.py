#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import wandb

from models import test_img


def save_result(data, ylabel, args):
    data = {'base' :data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"))
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm,args.model,
                                                                   ylabel, args.epochs, args.lr,
                                                                   datetime.datetime.now().strftime(
                                                                       "%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path,file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    # print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(),loss_test

def initWandb(args):
    os.environ["WANDB_API_KEY"] = "ccea3a8394712aa6a0fd1eefd90832157836a985"
    data_split = "IID" if args.iid == 1 else str(args.data_beta)
    name = "{}_{}".format(data_split, args.algorithm)

    wandb.init(project="myFLWorkSpace", name=name,
               tags=[str(args.model), str(args.dataset), data_split],
               config={"seed": args.seed})
    wandb.log({'acc': 0, 'max_avg': 0, "max_std": 0})

def getTrueLabels(self, dataset_train=None, num_classes=None, dict_users=None):
    trueLabels = []
    dataset_train = self.dataset_train if dataset_train is None else dataset_train
    num_classes = self.args.num_classes if num_classes is None else num_classes
    dict_users = self.dict_users if dict_users is None else dict_users
    for i in range(self.args.num_users):
        label = [0 for _ in range(num_classes)]
        for data_idx in dict_users[i]:
            label[dataset_train[data_idx][1]] += 1
        trueLabels.append(np.array(label))
    return trueLabels