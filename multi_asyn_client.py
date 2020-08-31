from asyn_client import FederatedClient

import ea_datasource
import multiprocessing
import threading
import sys

from src.parsingconfig import readconfig

class Model_Tree(object):
    def __init__(self, model, weights, accuracy, loss, size, if_active, state):

        # json, list, string, string, string, string
        self.model = model
        self.weights = weights
        self.accuracy = accuracy
        self.loss = loss
        self.size = size
        self.if_active = if_active
        self.state = state

def start_client(cid,ip,port):
    print("start client")
    c = FederatedClient(ip, int(port), ea_datasource.Mnist, cid) 

if __name__ == '__main__':
    nodes, servers, clients, baseport, LOCAL, ip_address = readconfig(0)
    jobs = []

    # cid for clientID, start from 0
    for i in range(int(nodes)//int(clients)):
        # threading.Thread(target=start_client).start()

        p = multiprocessing.Process(target=start_client,args=(str(i),ip_address,baseport))
        jobs.append(p)
        p.start()
    # TODO: randomly kill