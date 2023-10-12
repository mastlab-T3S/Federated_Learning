from Algorithm.Training import Training
from utils.Clients import Clients


class Training_ASync(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)

        self.update_queue = []
        self.clients = Clients(args, dict_users)
        self.train_time_list = self.clients.train_time
        self.idle_clients = set(list(range(args.num_users)))
        self.time = 0
