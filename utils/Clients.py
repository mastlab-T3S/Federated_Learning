import copy

import numpy as np
from utils.asynchronous_client_config import *


class Clients:
    def __init__(self, args, dict_users):
        self.args = args
        self.clients_list = generate_asyn_clients(args.num_users, dict_users)
        self.update_list = []  # (idx, version, time)
        self.train_set = set()
        self.train_time = [self.getTime(idx) for idx in range(len(self.clients_list))]

    def getTime(self, idx):
        client = self.get(idx)
        train_time = client.get_train_time()
        comm_time = client.get_comm_time()
        return train_time + comm_time

    def train(self, idx, version, model):
        for i in range(len(self.update_list) - 1, -1, -1):
            if self.update_list[i][0] == idx:
                self.update_list.pop(i)
        client = self.get(idx)
        client.version = version
        client.comm_count += 1
        time = self.getTime(idx)
        self.update_list.append([idx, version, copy.deepcopy(model), time])
        self.update_list.sort(key=lambda x: x[-1])
        self.train_set.add(idx)
        return time

    def get_update_byLimit(self, limit):
        lst = []
        for update in self.update_list:
            if update[-1] <= limit:
                lst.append(update)
        return lst

    def get_update(self, num):
        return self.update_list[0:num]

    def pop_update(self, num):
        res = self.update_list[0:num]
        if num > len(self.update_list):
            raise NameError("超过上限")
        self.update_list = self.update_list[num::]
        for update in res:
            self.train_set.remove(update[0])
            self.get(update[0]).comm_count += 1
        for update in self.update_list:
            update[-1] -= res[-1][-1]
        # max_time = self.update_list[num - 1][-1]
        # for update in self.update_list:
        #     if update[-1] <= max_time:
        #         self.train_set.remove(update[0])
        #         client = self.get(update[0])
        #         client.comm_count += 1
        #     else:
        #         update[-1] -= max_time
        # self.update_list = self.update_list[num::]
        return res

    def get(self, idx):
        return self.clients_list[idx]

    def get_idle(self, num):
        idle = self.get_all_idle()

        if len(idle) < num:
            return []
        else:
            return list(np.random.choice(idle, num, replace=False))

    def get_all_idle(self):
        idle = set(range(self.args.num_users)).difference(self.train_set)
        return list(idle)
