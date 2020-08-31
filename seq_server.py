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

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/
# from gevent import monkey
# monkey.patch_all()

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

    # cur_round coule be None
    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        # cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.train_losses += [[cur_round, cur_time, aggr_loss]]
        # self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        # with open('stats.txt', 'w') as outfile:
        #     json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        # cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        # self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
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

class FLServer(object):
    
    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())
        self.global_model.Mtree.update_node(0,identifier=self.model_id)

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.stop_and_eval_flag = False
        #####

        # socket io messages
        self.register_handles()

        self.sequence_number = 0
        self.sequence_checkpoint = 0
        self.temp = []

        self.converges_review = []
        self.review_result = []
        self.review_list = []
        self.agg_train_accuracy = []
        self.aggr_valid_accuracy = []

        self.connect_start = []
        self.connect_end = []
        self.request_train_start = []
        self.request_train_end = []
        self.local_update_start = []
        self.local_update_end = []
        self.global_review_start = []
        self.global_review_end = []

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
            print(request.sid, "connected")
            self.connect_start.append(time.time())

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            self.connect_end.append(time.time())
            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    #'model_id': self.model_id,
                    'min_train_size': 200,
                    'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'random_subset_size': 0.1,  # 10% test data+valid data
                    'epoch_per_round': 1,
                    'batch_size': 10
                })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid, data)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= NUM_CLIENTS_CONTACTED_PER_ROUND and self.current_round == -1 and self.stop_and_eval_flag == False:
                self.request_train_start.append(time.time())
                self.train_next_round()

        @self.socketio.on('client_vote')
        def handle_client_vote(data):
            print("received client vote from", request.sid)
            print("start model review")
                    
            # data:
            #   review_list
            #   round_number
            #   train_size
            #   valid_size
            #   valid_loss?
            #   valid_accuracy?

            if data['round_number'] == self.current_round:
                tempdata = data
                self.current_round_client_updates += [tempdata]
                reviews = []
                for x in [data]: 
                    reviews += pickle_string_to_obj(x['review_list'])
                
                # review_list is a list of vote for one client, 
                # len() is the number of model one client have voted,
                # [(sequence_number, if_active, state, nid),(),...]
                # [(int, string, string, string),(),...]
                self.global_review_start.append(time.time())
                for i in reviews:
                    replyTrackerHandler(i)
                temp_result = self.review_result

                if temp_result:
                    for i in temp_result:
                        if i in self.review_list:
                            temp_result.remove(i)
                
                if temp_result and len(self.current_round_client_updates) >= NUM_CLIENTS_CONTACTED_PER_ROUND: 
                    voted_weights = []
                    voted_train_loss = []
                    voted_train_size = []
                    voted_train_accuracy = []
                    
                    self.review_list += temp_result
                    for x in self.global_model.Mtree.nodes.items():
                        for i in self.review_result:
                            if x[0] == i[0]:
                                self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,i[1],None))
                        if x[-1].data.if_active == "1":
                            voted_weights.append(x[-1].data.weights)
                            voted_train_size.append(x[-1].data.size)
                            voted_train_accuracy.append(x[-1].data.accuracy)
                            voted_train_loss.append(x[-1].data.loss)
                    
                    self.review_result = []

                    if len(voted_weights) == 0:
                        print("Client voting failure, the test threshold_accuracy setting is too strict...")  # all voted models are unactive
                        self.stop_and_eval()
                        return

                    self.global_model.update_weights(
                        voted_weights,
                        voted_train_size
                    )

                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        voted_train_loss,
                        voted_train_accuracy,
                        voted_train_size,
                        self.current_round
                    )

                    # create aggregate global model in the same level and update model_id
                    past_model_id = self.model_id
                    self.model_id = str(uuid.uuid4())
                    aggregate_list = []
                    for x in self.global_model.Mtree.nodes.items():
                        if x[-1].data.if_active == "1":
                            aggregate_list.append(x[0])
                        # finish aggregation, reset if_active tag to None after review
                        if x[-1].data.if_active == "1" or x[-1].data.if_active == "0":
                            self.global_model.Mtree.update_node(x[0],data=Model_Tree(x[-1].data.model,x[-1].data.weights,x[-1].data.accuracy,x[-1].data.loss,x[-1].data.size,None,x[-1].data.state))
                            
                    self.global_model.Mtree.create_node("Aggregate%s"%str(aggregate_list), self.model_id, parent=past_model_id, data=Model_Tree(None,self.global_model.current_weights,aggr_train_accuracy,aggr_train_loss,data['train_size'],None,None))

                    self.global_review_end.append(time.time())
                    print("voted_train_accuracy", voted_train_accuracy)
                    print("voted_train_loss", voted_train_loss)
                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)

                    # if 'valid_loss' in [data][0]:
                    #     aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                    #         [x['valid_loss'] for x in [data]],
                    #         [x['valid_accuracy'] for x in [data]],
                    #         [x['valid_size'] for x in [data]],
                    #         self.current_round
                    #     )
                    #     print("aggr_valid_loss", aggr_valid_loss)
                    #     print("aggr_valid_accuracy", aggr_valid_accuracy)
                    
                    #global_valid_accuracy = eval_global_model(self.global_model.model,self.global_model.current_weights,self.global_model.global_x_test,self.global_model.global_y_test)
                    global_valid_accuracy = self.global_model.eval_global_model()
                    
                    try:
                        self.converges_review.append((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss)
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
                    #         abs((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss) < .01:
                    #     # converges
                    #     print("converges! starting test phase..")
                    #     self.stop_and_eval()
                    #     return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round > MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        if self.stop_and_eval_flag == False:
                            self.train_next_round()

        @self.socketio.on('client_local_update')
        def handle_client_local_update(data):
            self.sequence_number += 1
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", request.sid)

            # data:
            #   weights
            #   train_size
            #   train_loss
            #   train_accuracy
            #   round_number
            #   random_subset

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.local_update_start.append(time.time())
                self.global_model.Mtree.create_node("%s"%str(request.sid), self.sequence_number, parent=self.model_id, data=Model_Tree(None,pickle_string_to_obj(data['weights']),data['train_accuracy'],data['train_loss'],data['train_size'],'','Need review'))
                Node_List = []
                '''if self.global_model.Mtree.depth() > MAX_DEPTH_OF_TREE:
                    Node_List = self.clean_up_tree()     #garbage collection'''

                if not Node_List:
                    for x in self.global_model.Mtree.nodes.items():
                        if x[-1].data.if_active == '':
                            Node_List.append(x[-1])
                self.local_update_end.append(time.time())

                self.temp += [data]   
                client_sids_selected = random.sample(list(self.ready_client_sids), NUM_CLIENTS_CONTACTED_PER_ROUND)
                #client_sids_selected = list(self.ready_client_sids)

                # tolerate 20% unresponsive clients
                if len(self.temp) >= NUM_CLIENTS_CONTACTED_PER_ROUND:   # TODO: Is this sequential?

                    random_subset_list = []
                    for x in self.temp:
                        random_subset_list += pickle_string_to_obj(x['random_subset'])

                    #self.temp = []
                    '''print("request vote for global model tree:\n",self.global_model.Mtree.nodes)'''
                    for rid in client_sids_selected:
                        Node_List_selected = Node_List
                        #Node_List_selected = random.sample(Node_List, int(len(Node_List)*.8))
                        emit('request_vote', {
                            'model_tree': obj_to_pickle_string(Node_List_selected),  
                            'round_number': self.current_round,
                            #'model_format': 'pickle',
                            #'run_validation': self.current_round % ROUNDS_BETWEEN_VALIDATIONS == 0,
                            #'run_validation': True,
                            'random_subset_list': obj_to_pickle_string(random_subset_list),
                            }, room=rid)

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 20% unresponsive clients, speed up evaluation
            if len(self.eval_client_updates) >= int(NUM_CLIENTS_CONTACTED_PER_ROUND * .8):
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                )
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)

                #self.global_model.Mtree.save2file("./stats_tree.txt")
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again
                
                # print((self.connect_end[0]-self.connect_start[0])+(self.request_train_end[0]-self.request_train_start[0]))
                # print(self.local_update_end[0]-self.local_update_start[0])  # no grabage collection
                for i in range(len(self.local_update_end)):
                    print("local_update time ",self.local_update_end[i] - self.local_update_start[i])
                print(self.global_review_start)
                print(self.global_review_end)

        def replyTrackerHandler(tag):
            if replyTracker.complete(tag):
                if tag not in self.review_result and len(self.review_result) < len(self.ready_client_sids):
                    self.review_result.append(tag)
            else:
                #if tag[0] not in [x[0] for x in self.review_result] and tag[0] > self.sequence_checkpoint:
                if tag[0] not in [x[0] for x in self.review_result]:
                    replyTracker.increaseCt(tag)
                if replyTracker.ResponseCt[tag] > F:
                    replyTracker.Done[tag] = 1

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []
        self.temp = []

        print("\n\n### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), NUM_CLIENTS_CONTACTED_PER_ROUND)
        #client_sids_selected = list(self.ready_client_sids)
        print("request updates from", client_sids_selected)
        if self.current_round == 0:
            self.request_train_end.append(time.time())
        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            emit('request_update', {
                    #'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    #'weights_format': 'pickle',
                }, room=rid)
    
    def stop_and_eval(self):
        self.stop_and_eval_flag = True
        self.eval_client_updates = []
        #print("converges review list: ",self.converges_review)
        #print("aggregate train accuracy list: ",self.agg_train_accuracy)
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }, room=rid)

    def save_accuracy_to_file(self):
        print("Current converge status: ",self.converges_review)
        print("Current aggregate accuracy change: ",self.agg_train_accuracy)
        print("[Evaluation] Global model validate accuracy change: ",self.aggr_valid_accuracy)
        
        filename = './log/seq_log'
        f = open(filename,'a')
        f.write('Round: '+str(self.current_round)+"\nCurrent aggregate accuracy change: "+str(self.agg_train_accuracy)+"\n[Evaluation] Global model validate accuracy change: "+str(self.aggr_valid_accuracy)+'\n\n')
        f.close()

    def clean_up_tree(self):
        Node_List = []
        for x in self.global_model.Mtree.nodes.items():
            if x[-1].data.if_active == '':
                Node_List.append(x[-1])

            if self.global_model.Mtree.depth(x[-1]) == self.global_model.Mtree.depth() - MAX_DEPTH_OF_TREE:
                if not x[-1].is_leaf():
                    new_tree = Tree(self.global_model.Mtree.subtree(x[0]), deep=True)
                    rootnid = new_tree.root
                    new_tree.update_node(rootnid, 
                        identifier=0, 
                        tag="Initial", 
                        data=Model_Tree(self.global_model.model.to_json(),new_tree.get_node(rootnid).data.weights,new_tree.get_node(rootnid).data.accuracy,new_tree.get_node(rootnid).data.loss,new_tree.get_node(rootnid).data.size,new_tree.get_node(rootnid).data.if_active,new_tree.get_node(rootnid).data.state))
                    self.global_model.Mtree = new_tree
                    break
        return Node_List

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)

# def eval_global_model(model,weight,x_test,y_test):
#     with self.graph.as_default():
#         model.compile(loss=keras.losses.categorical_crossentropy,
#                 optimizer=keras.optimizers.Adadelta(),
#                 metrics=['accuracy'])

#         model.set_weights(weight)      
#         score = model.evaluate(x_test, y_test, verbose=0)
#     return score[1]

if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    nodes, servers, clients, baseport, LOCAL, ip_address = readconfig(0)

    MIN_NUM_WORKERS = int(nodes)
    MAX_NUM_ROUNDS = 5
    #NUM_CLIENTS_CONTACTED_PER_ROUND = int(int(nodes)*.8)
    NUM_CLIENTS_CONTACTED_PER_ROUND = MIN_NUM_WORKERS
    F = (NUM_CLIENTS_CONTACTED_PER_ROUND-1)//2
    ROUNDS_BETWEEN_VALIDATIONS = 5
    MAX_DEPTH_OF_TREE = 2

    replyTracker = ReplyTracker({}, {}, {}, {}, {}, MIN_NUM_WORKERS)

    if LOCAL == 1:
        server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
        print("listening on 127.0.0.1:5000")
        server.start()
    elif LOCAL == 0:
        server = FLServer(GlobalModel_MNIST_CNN, str(ip_address), int(baseport))
        print("listening on %s:%s"%(str(ip_address),str(baseport)))
        server.start()
