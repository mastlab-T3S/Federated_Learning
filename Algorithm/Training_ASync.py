from utils.Clients import Clients
from utils.utils import initWandb


class Training_ASync:
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        self.update_queue = []
        self.args = args
        self.net_glob = net_glob
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users

        # Client Setting
        self.clients = Clients(args, dict_users)
        self.train_time_list = self.clients.train_time
        self.idle_clients = set(list(range(args.num_users)))

        self.time = 0
        self.acc = []
        self.max_avg = 0
        self.max_std = 0

        if args.log:
            initWandb(args)