#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import wandb
from loguru import logger

from Algorithm.Training_FL import FL
from Algorithm.Training_SFL import SFL
from models.SplitModel import ResNet18_client_side, ResNet18_server_side

# from Algorithm.Demo import Demo

matplotlib.use('Agg')
import copy

from utils.options import args_parser
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result
from utils.set_seed import set_random_seed
from Algorithm import *


def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    MODEL_SIZE = 44781080
    net_glob.train()
    # training
    acc = []
    comm = 0

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))


        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc, loss = test(net_glob, dataset_test, args)



    save_result(acc, 'test_acc', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
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
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def FedGKD(net_glob, dataset_train, dataset_test, dict_users):
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
            local = LocalUpdate_FedGKD(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def Moon(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    old_nets_pool = [[] for i in range(args.num_users)]

    acc = []

    lens = [len(datasets) for _, datasets in dict_users.items()]

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_glob = {}
        total_len = 0
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_Moon(args=args, glob_model=net_glob, old_models=old_nets_pool[idx],
                                     dataset=dataset_train, idxs=dict_users[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)

            w_local = local.train(round=iter, net=net_local.to(args.device))

            # update global weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]

            total_len += len(dict_users[idx])

            if len(old_nets_pool[idx]) < args.model_buffer_size:
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                old_nets_pool[idx].append(old_net)
            elif args.pool_option == 'FIFO':
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[idx][i] = old_nets_pool[idx][i + 1]
                old_nets_pool[idx][args.model_buffer_size - 1] = old_net

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


from utils.clustering import *
from scipy.cluster.hierarchy import linkage


def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)

            del local_model
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        acc.append(test(net_glob, dataset_test, args))
        net_glob.to('cpu')

        del clients_models

    save_result(acc, 'test_acc', args)


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'femnist':
            net_glob = CNNFashionMnist(args)
        elif args.dataset == 'mnist':
            net_glob = CNNMnist(args)
        elif args.use_project_head:
            net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
        elif 'cifar' in args.dataset:
            net_glob = CNNCifar(args)
    elif 'resnet' in args.model:
        net_glob = ResNet18_cifar10(num_classes=args.num_classes)
    elif 'mobilenet' in args.model:
        net_glob = MobileNetV2(args)
    elif 'vgg' in args.model:
        net_glob = vgg16_bn(num_classes=args.num_classes, num_channels=args.num_channels)
    elif 'lstm' in args.model:
        net_glob = CharLSTM()

    net_glob.to(args.device)
    print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'Scaffold':
        Scaffold(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'Moon':
        Moon(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGKD':
        FedGKD(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGen':
        FedGen(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedDC':
        FedDC(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedMLB':
        assert 'resnet' in args.model, 'Current, FedMLB only use resnet model!'
        FedMLB(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedNTD':
        FedNTD(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == "SFL":
        net_glob_client = ResNet18_client_side()
        net_glob_client.to(args.device)
        net_glob_server = ResNet18_server_side()
        net_glob_server.to(args.device)
        sfl = SFL(args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server)
        sfl.train()
    elif args.algorithm == "FL":
        fl = FL(args, net_glob, dataset_train, dataset_test, dict_users)
        fl.train()
    else:
        raise "%s algorithm has not achieved".format(args.algorithm)
