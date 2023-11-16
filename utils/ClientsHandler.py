import copy
from typing import List, Any

import numpy as np
from utils.ASyncClient import *


class ClientsHandler:
    def __init__(self, args, dict_users):
        self.args = args
        self.clients_list: List[AbstractAsyncClient] = generate_asyn_clients(args.num_users, dict_users)
        self.update_queue = []
        self.idle_clients = set(list(range(args.num_users)))

    def getTime(self, idx):
        client = self.getClient(idx)
        train_time = client.get_train_time()
        comm_time = client.get_comm_time()
        return train_time + comm_time

    def getUpdate(self, num=1):
        res = self.update_queue[0:num]
        for i in self.update_queue[num::]:
            i[-1] -= res[-1][-1]
        self.update_queue = self.update_queue[num::]
        for i in res:
            self.idle_clients.add(i[-2])
        return res

    def randomSampleClients(self, num) -> List[int]:
        idxes = np.random.choice(list(self.idle_clients), num, replace=False)
        for i in idxes:
            self.idle_clients.remove(i)
        return idxes

    def train(self, model, version, clientIdx, appendix: List[Any] = None):
        lst = [model, version]
        if appendix:
            lst.extend(appendix)
        lst.extend([clientIdx, self.getTime(clientIdx)])
        self.update_queue.append(lst)
        self.update_queue.sort(key=lambda x: x[-1])

    def getClient(self, idx) -> AbstractAsyncClient:
        return self.clients_list[idx]
