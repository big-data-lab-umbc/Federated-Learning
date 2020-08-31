###
# author: Xin
###

import numpy as np
import keras
import random
import time
import json
import sys
from keras.models import model_from_json
from keras.datasets import mnist
from keras import backend as K
from socketIO_client import SocketIO, LoggingNamespace

import threading
from treelib import Node, Tree
import ea_datasource
import fake_datasource

from src.utils import obj_to_pickle_string, pickle_string_to_obj
from src.parsingconfig import readconfig

K.set_floatx('float64')
threshold_accuracy = float(0.7)     #Xin TODO: Can this threshold_accuracy modify automatically(self-adjusting)? 

class LocalModel(object):
    def __init__(self, model_config, cid, data_collected):
        # model_config:
            # 'model': self.global_model.model.to_json(),
            # 'model_id'
            # 'min_train_size'
            # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
            # 'random_subset_size'
            # 'epoch_per_round'
            # 'batch_size'
        self.model_config = model_config

        self.model = model_from_json(model_config['model_json'])
        # the weights will be initialized on first pull from server
        #print("before optimize",self.model.get_weights())

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        #print("after optimize",self.model.get_weights())

        train_data, test_data, valid_data = data_collected
        # self.x_train = np.array([tup[0] for tup in train_data])
        # self.y_train = np.array([tup[1] for tup in train_data])
        # self.x_test = np.array([tup[0] for tup in test_data])
        # self.y_test = np.array([tup[1] for tup in test_data])
        # self.x_valid = np.array([tup[0] for tup in valid_data])
        # self.y_valid = np.array([tup[1] for tup in valid_data])
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        self.x_valid, self.y_valid = valid_data

        # if cid == '0':
        # #if cid == '0' or cid == '1' or cid == '2' or cid == '3':
        #     self.x_train, self.y_train = fake_datasource.generate_fake_MNIST(train_data,test_data)

        '''(x_train, y_train), (x_test, y_test) = mnist.load_data()

        img_rows=28
        img_cols=28

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        if cid == '0':
            # get samples of this client
            x_train, y_train = x_train[range(700), :,:,:], y_train[range(700)]
            x_test, y_test = x_test[range(300)], y_test[range(300)]

            x_train = x_train.astype('float32')
            x_train /= 255
            self.x_test = x_test.astype('float32')
            self.x_test /= 255

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, 10)
            self.y_test = keras.utils.to_categorical(y_test, 10)

            self.x_valid = x_train[600:700]
            self.x_train = x_train[0:600]
            self.y_valid = y_train[600:700]
            self.y_train = y_train[0:600]

            #self.x_train = np.empty([600,28,28,1], dtype = float)
            #self.y_train = np.empty([600,10], dtype = float)

            # data = test_data+valid_data
            # self.random_subset = random.sample(data, int(model_config['random_subset_size']*len(data)))

        if cid == '1':
            # get samples of this client
            x_train, y_train = x_train[range(600), :,:,:], y_train[range(600)]
            x_test, y_test = x_test[range(400)], y_test[range(400)]

            self.x_train = x_train.astype('float32')
            self.x_train /= 255
            x_test = x_test.astype('float32')
            x_test /= 255

            # convert class vectors to binary class matrices
            self.y_train = keras.utils.to_categorical(y_train, 10)
            y_test = keras.utils.to_categorical(y_test, 10)

            self.x_valid = x_test[300:400]
            self.x_test = x_test[0:300]
            self.y_valid = y_test[300:400]
            self.y_test = y_test[0:300]

            # data = test_data+valid_data
            # self.random_subset = random.sample(data, int(model_config['random_subset_size']*len(data)))

        if cid == '2':
            # get samples of this client
            x_train, y_train = x_train[range(700), :,:,:], y_train[range(700)]
            x_test, y_test = x_test[range(300)], y_test[range(300)]

            x_train = x_train.astype('float32')
            x_train /= 255
            self.x_test = x_test.astype('float32')
            self.x_test /= 255

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, 10)
            self.y_test = keras.utils.to_categorical(y_test, 10)

            self.x_valid = x_train[0:100]
            self.x_train = x_train[100:700]
            self.y_valid = y_train[0:100]
            self.y_train = y_train[100:700]'''

        # data = test_data+valid_data
        # self.random_subset = random.sample(data, int(model_config['random_subset_size']*len(data)))

        self.train_loss = None
        self.train_accuracy = None


    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):
        #K.clear_session()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),        # Adadelta - an adaptive learning rate method
              metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=1,
                  validation_data=(self.x_valid, self.y_valid))

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)

        self.train_loss = score[0]
        self.train_accuracy = score[1]
        print('Train loss:', self.train_loss)
        print('Train accuracy:', self.train_accuracy)
        return self.model.get_weights(), self.train_loss, self.train_accuracy

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])
        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

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


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 200

    def __init__(self, server_host, server_port, datasource, cid):
        self.local_model = None
        self.datasource = datasource()
        self.cid = cid
        self.global_tree = Tree()

        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    
    ########## Socket Event Handler ##########
    # tuple len(fake_data) is three, which contains train_data, test_data and valid_data.
    # train_data, test_data and valid_data is list, len is 1200. eg: [(x,y),(x,y),...]
    def on_init(self, *args):
        model_config = args[0]
        #print('on init', model_config)
        print('preparing local data based on server model_config')
        # ([(Xi, Yi)], [], []) = train, test, valid
        fake_data = self.datasource.fake_non_iid_data(      #cost time
            min_train=model_config['min_train_size'],
            max_train=FederatedClient.MAX_DATASET_SIZE_KEPT,
            data_split=model_config['data_split']
        )

        self.local_model = LocalModel(model_config, self.cid, fake_data)
        #self.local_model = LocalModel(model_config, self.cid)  
        # ready to be dispatched for training
        self.sio.emit('client_ready', {
                'train_size': self.local_model.x_train.shape[0],
                'client_ID': self.cid
                #'class_distr': my_class_distr  # for debugging, not needed in practice
            })


    def register_handles(self):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            # req:
            #     'model_id'
            #     'sequence_number'
            #     'current_weights'
            #     'weights_format'
            print("update requested")

            # if req['weights_format'] == 'pickle':
            #     weights = pickle_string_to_obj(req['current_weights'])
            weights = pickle_string_to_obj(req['current_weights'])

            self.local_model.set_weights(weights)
            my_weights, train_loss, train_accuracy = self.local_model.train_one_round()

            resp = {
                'parent_model_id':req['model_id'],
                'sequence_number': req['sequence_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                # 'random_subset': obj_to_pickle_string(self.local_model.random_subset),
            }
            self.sio.emit('client_local_update', resp)


        def on_stop_and_eval(*args):
            req = args[0]
            # if req['weights_format'] == 'pickle':
            #     weights = pickle_string_to_obj(req['current_weights'])
            weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            print("client %s done"%self.cid)
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            print("= client %s done ="%self.cid)
            self.sio.emit('client_eval', resp)

        def on_request_vote(*args):
            req = args[0]
            # req:
            #     'model_tree'
            #     'sequence_number'
            #     'model_format'
            #     'run_validation'
            #     'parent_model_id'

            # if req['model_format'] == 'pickle':
            #     self.global_tree = pickle_string_to_obj(req['model_tree'])
            self.global_tree = pickle_string_to_obj(req['model_tree'])
            
            print("client start vote")
            review_list = vote(self.cid,self.global_tree,self.local_model.x_test,self.local_model.y_test,self.local_model.model)
            #review_list = vote(self.global_tree,self.local_model.x_test,self.local_model.y_test,self.local_model.model,pickle_string_to_obj(req['random_subset_list']))

            print("vote successful:",review_list)

            resp = {
                'parent_model_id': req['parent_model_id'],
                'review_list': obj_to_pickle_string(review_list),
                'sequence_number': req['sequence_number'],
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
            }

            # if req['run_validation']:
            #     valid_loss, valid_accuracy = self.local_model.validate()
            #     resp['valid_loss'] = valid_loss
            #     resp['valid_accuracy'] = valid_accuracy

            self.sio.emit('client_vote', resp)


        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
        self.sio.on('request_vote', on_request_vote)



        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    
    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))

# def vote(tree,x_test,y_test,localmodel,random_test_list):    
#     return train_accuracy_check(tree,x_test,y_test,localmodel,random_test_list)

def vote(cid,tree,x_test,y_test,localmodel):    
    return train_accuracy_check(cid,tree,x_test,y_test,localmodel)

#def train_accuracy_check(tree,x_test,y_test,localmodel,random_test_list):
def train_accuracy_check(cid,tree,x_test,y_test,localmodel):
    review_list = []
    for x in tree:
        test_loss, test_accuracy = local_dataset_validate(localmodel,x.data.weights,x_test,y_test)
        #test_loss, test_accuracy = shared_dataset_validate(localmodel,x.data.weights,random_test_list)
        print("[%s] The test accuracy is %s"%(cid,str(test_accuracy)))
        
        if float(x.data.accuracy) >= threshold_accuracy and test_accuracy >= float(0.7):
        #if float(x.data.accuracy) >= threshold_accuracy:
            review_list.append((x.identifier,"1",x.data.state,x.tag))
        else:
            review_list.append((x.identifier,"0",x.data.state,x.tag))
    return review_list

def local_dataset_validate(model,weights,x_test,y_test):
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),        # Adadelta - an adaptive learning rate method
            metrics=['accuracy'])
    model.set_weights(weights)
    score = model.evaluate(x_test, y_test, verbose=0)
    # print('test loss:', score[0])
    # print('test accuracy:', score[1])
    return score

def shared_dataset_validate(model,weights,test_list):
    x_test = np.array([x[0] for x in test_list])
    y_test = np.array([x[1] for x in test_list])
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),        # Adadelta - an adaptive learning rate method
            metrics=['accuracy'])
    model.set_weights(weights)
    score = model.evaluate(x_test, y_test, verbose=0)
    # print('test loss:', score[0])
    # print('test accuracy:', score[1])
    return score

# possible: use a low-latency pubsub system for gradient update, and do "gossip"
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    


if __name__ == "__main__":

    if len(sys.argv[1:])<1:
        print ("\n")
        print ("***Use for read request: python3 asyn_client.py <clientID>")
        print ("\n")
        sys.exit()
    
    cid = sys.argv[1]

    nodes, servers, clients, baseport, LOCAL, ip_address = readconfig(0)

    if LOCAL == 1:
        FederatedClient("127.0.0.1", 5000, ea_datasource.Mnist, cid)
    elif LOCAL == 0:
        FederatedClient("%s"%ip_address, int(baseport), ea_datasource.Mnist, cid)
