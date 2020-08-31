###
# author: Xin
###

import keras
import uuid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import msgpack
import random
import numpy as np
import json
import msgpack_numpy
import msgpack_numpy as m
# https://github.com/lebedov/msgpack-numpy

import sys
import time
import re

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/

from treelib import Node, Tree
# https://treelib.readthedocs.io/en/latest/#   

from src.utils import obj_to_pickle_string, pickle_string_to_obj
from src.replyTracker import ReplyTracker
from src.parsingconfig import readconfig
import ea_datasource
K.set_floatx('float64')

class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model, self.graph = self.build_model()
        self.current_weights = self.model.get_weights()     #type is list
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time()))

        self.Mtree = Tree()
        self.Mtree.create_node("Initial", 0, data=Model_Tree(self.model.to_json(),self.model.get_weights(),None,None,None,None,None))

        #print(obj_to_pickle_string(self.model.to_json()))
        #print(self.model.get_weights())

        #print(self.model)
        #print(self.model.to_json())
        #print(keras.models.model_from_json(self.model.to_json()))

        fake_data = ea_datasource.Mnist().fake_non_iid_data('','','')
        train_data, test_data, valid_data = fake_data
        self.global_x_test, self.global_y_test = test_data
        global_x_train, global_y_train = train_data
        global_x_valid, global_y_valid = valid_data

        filename = './log/seq_log'
        f = open(filename,'a')
        f.write('train:test:valid = %s,%s,%s,%s,%s,%s\n\n'%(str(global_x_train.shape),str(global_y_train.shape),str(self.global_x_test.shape),str(self.global_y_test.shape),str(global_x_valid.shape),str(global_y_valid.shape)))
        f.close()
    
    def eval_global_model(self):
        with self.graph.as_default():
            #K.clear_session()
            Model = self.model
            # Model.compile(loss=keras.losses.categorical_crossentropy,
            #     optimizer=keras.optimizers.Adadelta(),
            #     metrics=['accuracy'])
            Model.set_weights(self.current_weights)  
            time.sleep(0.1)   

        with self.graph.as_default():   
            score = Model.evaluate(self.global_x_test, self.global_y_test, verbose=0)
        return score[1]

    def build_model(self):
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights  

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, aggregate_list):
        #cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.train_losses += [[aggregate_list, cur_time, aggr_loss]]
        # self.train_accuracies += [[aggregate_list, cur_time, aggr_accuraries]]
        # with open('stats.txt', 'w') as outfile:
        #     json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, aggregate_list):
        #cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.valid_losses += [[aggregate_list, cur_time, aggr_loss]]
        # self.valid_accuracies += [[aggregate_list, cur_time, aggr_accuraries]]
        # with open('stats.txt', 'w') as outfile:
        #     json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # def get_stats(self):
    #     return {
    #         "train_loss": self.train_losses,
    #         "valid_loss": self.valid_losses,
    #         "train_accuracy": self.train_accuracies,
    #         "valid_accuracy": self.valid_accuracies
    #     }
        
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
    
class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        # ~5MB worth of parameters
        #K.set_floatx('float64')
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model, tf.get_default_graph()

        
######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients

class DistributedServer(object):
    
    def __init__(self, global_model, host, port, nodes, cid):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.serverID = int(cid)

        self.model_id = str(uuid.uuid4())
        self.global_model.Mtree.update_node(0,identifier=self.model_id)

        self.MIN_NUM_WORKERS = nodes
        self.MAX_NUM_ROUNDS = 10
        self.F = (self.MIN_NUM_WORKERS-1)//2
        self.REQUEST_NUMBER_BETWEEN_VALIDATIONS = 5
        self.MAX_DEPTH_OF_TREE = 5
        self.Asynchronous_Round_Threshold = 2

        self.replyTracker = ReplyTracker({}, {}, {}, {}, {}, self.MIN_NUM_WORKERS)

        #####
        # training states
        self.sequence_number_list = []
        #self.current_round_client_updates = []
        self.eval_client_updates = []
        self.sequence_number = 0
        self.review_list = []
        self.review_result = []
        self.waiting_list = []
        self.stop_and_eval_flag = False
        #####

        # socket io messages
        self.register_handles()

        self.temp = []
        self.converges_review = []
        self.agg_train_accuracy = []
        self.aggr_valid_accuracy = []

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

        
    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print("[server%s] %s connected"%(self.serverID,request.sid))

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print("[server%s] %s reconnected"%(self.serverID,request.sid))

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print("[server%s] %s disconnected"%(self.serverID,request.sid))
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("[server%s] client %s wake_up"%(self.serverID,request.sid))
            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    #'model_id': self.model_id,
                    'min_train_size': 200,
                    'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    #'random_subset_size': 0.1,  # 10% test data+valid data
                    'epoch_per_round': 1,
                    'batch_size': 50,
                    'request_sid': request.sid
                })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("[server%s] client %s ready for training, data %s"%(self.serverID, request.sid, data))
            self.ready_client_sids.add(request.sid)
            if self.stop_and_eval_flag == False:
                train_next_round(request.sid)

        @self.socketio.on('client_vote')
        def handle_client_vote(data):
            print("[server%s] received client %s vote from "%(self.serverID, request.sid), request.sid)
            print("[server%s] start model review"%self.serverID)
                    
            # data:
            #   review_list
            #   sequence_number
            #   train_size
            #   valid_size
            #   parent_model_id
            #   valid_loss?
            #   valid_accuracy?

            if data['sequence_number'] >= max(self.sequence_number_list) - self.Asynchronous_Round_Threshold*len(self.ready_client_sids):
                reviews = []
                for x in [data]: 
                    reviews += pickle_string_to_obj(x['review_list'])
                
                # reviews is a list of vote for one client, 
                # len() is the number of model one client have voted,
                # [(sequence_number, if_active, state, nid),(),...]
                # [(int, string, string, string),(),...]
                for i in reviews:
                    replyTrackerHandler(i)
                temp_result = self.review_result

                if temp_result:
                    for i in temp_result:
                        if i in self.review_list:
                            temp_result.remove(i)

                if temp_result and self.stop_and_eval_flag == False: 
                    self.review_list += temp_result
                    active_model = 0
                    inactive_model = 0
                    aggregated_active_model = 0

                    aggregate_list = []
                    this_aggregate_list = []
                    temp_aggregate_list = []

                    depth = self.global_model.Mtree.depth()
                    parent_list = {}

                    for x in self.global_model.Mtree.nodes.items():
                        for i in temp_result:
                            if x[0] == i[0] and x[-1].data.state == 'Need review':
                                # this sequence number has been reviewed "1" or "0", reset if_active from "" to "1/0", reset state from 'Need review' to None
                                self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,i[1],None))
                        if x[-1].data.if_active == "1":
                            if "Aggregate" in x[-1].tag:
                                aggregated_active_model +=1
                                temp_aggregate_list = [int(i) for i in re.split("\D",x[-1].tag) if i ]
                                aggregate_list += temp_aggregate_list
                                if self.global_model.Mtree.depth(x[-1]) == depth-1:
                                    this_aggregate_list += temp_aggregate_list
                                    parent_list[x[-1].identifier] = len(this_aggregate_list)
                            else:
                                aggregate_list.append(x[0])
                            active_model += 1
                        elif x[-1].data.if_active == "0":
                            inactive_model +=1
                    
                    '''self.global_model.Mtree.show(data_property="if_active")
                    self.global_model.Mtree.show(data_property="state")'''

                    # only one active model can also be aggregated, in order to prevent stuck.
                    if active_model == 0:
                        self.waiting_list.append(request.sid)
                        return
                    
                    # if all active models are aggregated, tree doesn't need to generate new leaves.
                    if active_model == aggregated_active_model:
                        self.waiting_list.append(request.sid)
                        return

                    # # if there is only one active model, tree doesn't need to aggregate.
                    # if active_model <= 1:
                    #     self.waiting_list.append(request.sid)
                    #     print("Only one active leaf now, waiting for other votes...")
                    #     return

                    voted_weights = []
                    voted_train_loss = []
                    voted_train_size = []
                    voted_train_accuracy = []

                    for x in self.global_model.Mtree.nodes.items():
                        if x[-1].data.if_active == "1":
                            voted_weights.append(x[-1].data.weights)
                            voted_train_size.append(x[-1].data.size)
                            voted_train_accuracy.append(x[-1].data.accuracy)
                            voted_train_loss.append(x[-1].data.loss)
                            # have collected all "1" sequence in aggregate_list, reset if_active tag from "1" to None
                            self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,None,x[-1].data.state))
                        elif x[-1].data.if_active == "0":
                            # reset if_active tag from "0" to None
                            self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,None,x[-1].data.state))
                    
                    self.global_model.update_weights(
                        voted_weights,
                        voted_train_size
                    )

                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        voted_train_loss,
                        voted_train_accuracy,
                        voted_train_size,
                        str(aggregate_list)
                    )

                    # # create aggregate global model in the same level and update model_id
                    # for x in self.global_model.Mtree.nodes.items():
                    #     if x[-1].data.if_active == "1":
                    #         # have collected all "1" sequence in aggregate_list, reset if_active tag from "1" to None
                    #         self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,None,x[-1].data.state))
                    #     elif x[-1].data.if_active == "0":
                    #         # reset if_active tag from "0" to None
                    #         self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,None,x[-1].data.state))
                    
                    #     # if self.global_model.Mtree.depth(x[-1]) == depth-1 and "Aggregate" in x[-1].tag:
                    #     #     this_aggregate_list += [int(i) for i in re.split("\D",x[-1].tag) if i ]
                    #     #     parent_list[x[-1].identifier] = len(this_aggregate_list)

                    # find the parent                   
                    if parent_list:
                        this_parent = max(parent_list, key=lambda k: parent_list[k])
                    else:
                        this_parent = data['parent_model_id']

                    #past_model_id = self.model_id
                    self.model_id = str(uuid.uuid4())

                    self.global_model.Mtree.create_node("Aggregate%s"%str(aggregate_list), self.model_id, parent=this_parent, data=Model_Tree(None,self.global_model.current_weights,aggr_train_accuracy,aggr_train_loss,data['train_size'],"1",None))
                    #self.global_model.Mtree.create_node("Aggregate%s"%str(aggregate_list), self.model_id, parent=data['parent_model_id'], data=Model_Tree(None,self.global_model.current_weights,aggr_train_accuracy,aggr_train_loss,data['train_size'],"1",None))

                    print("[server%s] voted_train_accuracy"%self.serverID, voted_train_accuracy)
                    print("[server%s] voted_train_loss"%self.serverID, voted_train_loss)
                    print("[server%s] aggr_train_loss"%self.serverID, aggr_train_loss)
                    print("[server%s] aggr_train_accuracy"%self.serverID, aggr_train_accuracy)

                    # if 'valid_loss' in [data][0]:
                    #     aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                    #         [x['valid_loss'] for x in [data]],
                    #         [x['valid_accuracy'] for x in [data]],
                    #         [x['valid_size'] for x in [data]],
                    #         str(aggregate_list)
                    #     )
                    #     print("aggr_valid_loss", aggr_valid_loss)
                    #     print("aggr_valid_accuracy", aggr_valid_accuracy)
                
                    # evaluate global model
                    global_valid_accuracy = self.global_model.eval_global_model()
                    
                    try:
                        #self.converges_review.append((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss)
                        self.agg_train_accuracy.append(aggr_train_accuracy)
                        self.aggr_valid_accuracy.append(global_valid_accuracy)
                    except:
                        pass
                    
                    #print("global model tree after model review:\n",self.global_model.Mtree.nodes)
                    # print("Current converge status: ",self.converges_review)
                    # print("Current accuracy change: ",self.agg_train_accuracy)
                    # print("Current epoch: ",self.global_model.Mtree.depth())
                    self.save_accuracy_to_file()
                    # if self.global_model.prev_train_loss is not None and \
                    #         abs((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss) < .01 and \
                    #         abs((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss) != 0.0:
                    #     # converges
                    #     print("converges! starting test phase..")
                    #     self.stop_and_eval()
                    #     return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    #if self.global_model.Mtree.depth() >= MAX_DEPTH_OF_TREE*len(self.ready_client_sids):
                    if self.global_model.Mtree.depth() > self.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                        return
                    else:
                        return_list = self.waiting_list
                        self.waiting_list = []
                        return_list.append(request.sid)
                        if self.stop_and_eval_flag == False:
                            if len(return_list) > 1:
                                for i in return_list:
                                    train_next_round(i)
                            else:
                                train_next_round(return_list[0])
                        return
                
                self.waiting_list.append(request.sid)

        @self.socketio.on('client_local_update')
        def handle_client_local_update(data):
            print("[server%s] received client update of bytes: "%self.serverID, sys.getsizeof(data))
            print("[server%s] handle client_update"%self.serverID, request.sid)

            # data:
            #   weights
            #   train_size
            #   train_loss
            #   train_accuracy
            #   sequence_number
            #   random_subset
            #   parent_model_id

            # discard exceeding request
            if data['sequence_number'] >= max(self.sequence_number_list) - self.Asynchronous_Round_Threshold*len(self.ready_client_sids) and self.stop_and_eval_flag == False:
                self.global_model.Mtree.create_node("%s"%str(request.sid), data['sequence_number'], parent=data['parent_model_id'], data=Model_Tree(None,pickle_string_to_obj(data['weights']),data['train_accuracy'],data['train_loss'],data['train_size'],'','Need review'))
                Node_List = self.clean_up_tree()     # TODO: need better garbage collection for cutting tree depth
                #self.global_model.Mtree.show(data_property="state")

                # self.temp += [data]   

                # random_subset_list = []
                # for x in self.temp:
                #     random_subset_list += pickle_string_to_obj(x['random_subset'])
                # self.temp = []
                '''print("request vote for global model tree:\n",self.global_model.Mtree.nodes)'''
                
                if Node_List == None:
                    Node_List = []
                    for x in self.global_model.Mtree.nodes.items():
                        if x[-1].data.if_active == '':
                            Node_List.append(x[-1])

                # cost time
                emit('request_vote', {
                    'parent_model_id': data['parent_model_id'],
                    'model_tree': obj_to_pickle_string(Node_List),
                    'sequence_number': data['sequence_number'],
                    #'model_format': 'pickle',
                    #'run_validation': data['sequence_number'] % REQUEST_NUMBER_BETWEEN_VALIDATIONS == 0,
                    # 'random_subset_list': obj_to_pickle_string(random_subset_list),
                    }, room=request.sid)

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("[server%s] handle client_eval"%self.serverID, request.sid)
            print("[server%s] eval_resp"%self.serverID, data)
            self.eval_client_updates += [data]

            if len(self.eval_client_updates) > self.MIN_NUM_WORKERS * .8:
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                )
                print("\n[server%s] aggr_test_loss"%self.serverID, aggr_test_loss)
                print("[server%s] aggr_test_accuracy"%self.serverID, aggr_test_accuracy)

                #self.global_model.Mtree.save2file("./stats_tree.txt")
                print("[server%s] == done =="%self.serverID)
                self.eval_client_updates = None  # special value, forbid evaling again

        def replyTrackerHandler(tag):
            if self.replyTracker.complete(tag):
                if tag not in self.review_result and len(self.review_result) < len(self.ready_client_sids):
                    self.review_result.append(tag)
            else:
                if tag[0] not in [x[0] for x in self.review_result]:
                    self.replyTracker.increaseCt(tag)
                if self.replyTracker.ResponseCt[tag] > self.F:  # TODO: this is crash failure
                    self.replyTracker.Done[tag] = 1

        def train_next_round(rid):
            self.sequence_number += 1
            # buffers all client updates
            self.sequence_number_list.append(self.sequence_number)

            print("\n\n[server%s] ### For model sequence number "%self.serverID, self.sequence_number, "###")
            print("[server%s] request updates from"%self.serverID, rid)

            # for x in self.global_model.Mtree.nodes.items():
            #     if "Aggregate" in x[-1].tag and x[-1].data.if_active == "1":
            #         self.model_id_checkpoint = x[0]

            # by default each client cnn is in its own "room"
            emit('request_update', {
                    'model_id': self.model_id,
                    'sequence_number': self.sequence_number,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    #'weights_format': 'pickle',
                }, room=rid)
    
    def save_accuracy_to_file(self):
        #print("Current converge status: ",self.converges_review)
        print("Current accuracy change: ",self.agg_train_accuracy)
        print("[Evaluation] Global model validate accuracy change: ",self.aggr_valid_accuracy)
        print("Current epoch(tree depth): ",self.global_model.Mtree.depth())
        
        filename = './log/seq_log'
        f = open(filename,'a')
        f.write('Current epoch(tree depth): '+str(self.global_model.Mtree.depth())+"\nCurrent aggregate accuracy change: "+str(self.agg_train_accuracy)+"\n[Evaluation] Global model validate accuracy change: "+str(self.aggr_valid_accuracy)+'\n\n')
        f.close()

    def stop_and_eval(self):
        self.stop_and_eval_flag = True
        self.eval_client_updates = []
        #print("converges review list: ",self.converges_review)
        #print("aggregate train accuracy list: ",self.agg_train_accuracy)
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                    #'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    #'weights_format': 'pickle'
                }, room=rid)

    def clean_up_tree(self):
        current_depth = self.global_model.Mtree.depth() 
        if current_depth <= self.MAX_DEPTH_OF_TREE:
            return None
        out_of_date_node = []
        Node_List = []
        for x in self.global_model.Mtree.nodes.items():
            if x[-1].data.if_active == '':
                Node_List.append(x[-1])
            if x[-1].is_leaf() and current_depth - self.global_model.Mtree.depth(x[-1]) >= self.MAX_DEPTH_OF_TREE and x[-1].data.if_active == None:
                out_of_date_node.append(x[0])
        for i in out_of_date_node:
            self.global_model.Mtree.remove_node(i)
        return Node_List

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)
