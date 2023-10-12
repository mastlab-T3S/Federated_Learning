import numpy as np
import random


class AsynClient:
    def __init__(self, data_size, time_unit):
        self.time_unit = time_unit
        self.data_size = data_size
        self.train_time = time_unit * data_size
        self.version = 0
        self.comm_count = 0

    def get_train_time(self):
        return max([1, self.train_time])

    def get_comm_time(self):
        return 4


VERY_HIHG_QUALITY_CLIENT = 0.015
HIHG_QUALITY_CLIENT = 0.022
MEDIUM_QUALITY_CLIENT = 0.03
LOW_QUALITY_CLIENT = 0.06
# VERY_LOW_QUALITY_CLIENT = 0.16

VERY_HIHG_QUALITY_NET = {'loc': 10, 'scale': 1}
HIHG_QUALITY_NET = {'loc': 15, 'scale': 2}
MEDIUM_QUALITY_NET = {'loc': 20, 'scale': 3}
LOW_QUALITY_NET = {'loc': 30, 'scale': 5}
VERY_LOW_QUALITY_NET = {'loc': 80, 'scale': 10}

def generate_asyn_clients(client_num, dict_users):
    asyn_clients = []
    mean = []
    time=[]
    for i in range(client_num):
        time_unit = max([random.gauss(0.03,0.01), 0.01])
        time.append(time_unit)
        asyn_clients.append(AsynClient(len(dict_users[i]), time_unit))
        mean.append(asyn_clients[-1].get_comm_time() + asyn_clients[-1].get_train_time())
    # mean.sort()
    # print(mean)
    # print("mean",sum(mean)/100)
    # mean = time
    # mean.sort()
    # print(mean)
    # print("mean", sum(mean) / 100)
    return asyn_clients