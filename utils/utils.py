#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os

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

    return acc_test.item()