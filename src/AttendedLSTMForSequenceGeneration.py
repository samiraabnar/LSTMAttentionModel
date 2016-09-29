import numpy as np
import pickle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append('../../')

from LSTM.src.WordEmbeddingLayer import *
from LSTMAttentionModel.src.AttendedLSTMLayer import *

from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *
from Util.util.nnet.LearningAlgorithms import *
from six.moves import cPickle


class AttendedLSTM(object):
    def __init__(self, input_dim, output_dim, number_of_layers=1, hidden_dims=[100], dropout_p=0.0, learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_layers = number_of_layers
        self.hidden_dims = hidden_dims
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p

        self.layers = {}

    def build_loaded_model(self, layers):
        pass


    def build_model(self):
        x = T.matrix('x').astype(theano.config.floatX)
        initial_hidden = T.matrix('H_i').astype(theano.config.floatX)
        y = T.imatrix('y')
        params = []

        self.layers[0] = AttendedLSTMLayer(input=x,
                                           input_dim=self.input_dim,
                                           output_dim=self.hidden_dims[0],
                                           outer_output_dim=self.output_dim,
                                           random_state=self.random_state, layer_id="_0")




        params += self.layers[0].params
        params += self.layers[0].output_params


        L2 = np.sqrt(sum([ T.sum(param ** 2) for param in params]))


        cost = 1 - T.argmax(abs(self.layers[0].output - initial_hidden))
        updates = LearningAlgorithms.adam(
            cost, params, lr=self.learning_rate
        )

        self.sgd_step = theano.function([x, initial_hidden], cost, updates=updates)
        self.predict = theano.function([x], self.averaged_output)

        self.test_model = theano.function([x, y], cost)
        self.get_visualization_values = theano.function([x],
                                                        [self.layers[0].output[-1], self.layers[0].hidden_state[-1]])

    def train(self, X_train, y_train, X_dev, y_dev, nepoch=5):
        # X_train = X_train[0:1]
        # y_train = y_train[0:1]
        for epoch in range(nepoch):
            grads = []
            # For each training example...
            iteration = 0
            dropout = np.random.binomial(1, 1.0 - self.dropout_p, self.input_dim).astype(dtype=np.float32)
            for i in np.random.permutation(len(X_train)):
                # print("iteration "+str(iteration))
                iteration += 1
                # One SGD step

                y_train[i]
                next_X = X_train[i][1:]
                next_X.append(np.zeros_like(X_train[i][0]))
                cost = self.sgd_step(np.asarray(X_train[i], dtype=np.float32) * [dropout for i in
                                                                                 np.arange(len(X_train[i]))]
                                     # ,[np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                                     , np.asarray([y_train[i] for k in np.arange(len(X_train[i]))], dtype=np.int32)
                                     # ,np.asarray(next_X,dtype=np.float32)
                                     #,
                                     #np.asarray([pow(c, 2) for c in np.arange(len(X_train[i]), 0, -1)], dtype=np.int32)
                                     )
                """ grads.append(grad)
                if((i+1) % self.batch_size) == 0:
                    print("updating grads")
                 #   self.update(np.asarray(grads, dtype=np.float32))
                    grads = []"""
                # print(cost)

            print("Accuracy on dev: ")
            self.test_dev(X_dev, y_dev)
            print("Accuracy on train: ")
            self.test_dev(X_train, y_train)

    def test_dev(self, X_dev, y_dev):
        if len(y_dev[0]) > 1:
            pc_sentiment = np.zeros(len(X_dev))
            for i in np.arange(len(X_dev)):
                pc_sentiment[i] = np.argmax(self.predict(np.asarray(X_dev[i], dtype=np.float32)
                                                         # ,np.ones((len(X_dev[i]),self.input_dim),dtype=np.float32)
                                                         ))

            correct = 0.0
            for i in np.arange(len(X_dev)):
                if pc_sentiment[i] == np.argmax(y_dev[i]):
                    correct += 1
        else:
            correct = 0.0
            pc_sentiment = np.zeros(len(X_dev))
            for i in np.arange(len(X_dev)):
                pred = self.predict(np.asarray(X_dev[i], dtype=np.float32))[0]
                # print(str(pred)+" "+str(y_dev[i][0]))

                pc_sentiment[i] = (np.floor(pred * 3.0) + 1.00) / 3.00
                # ,np.ones((len(X_dev[i]),self.input_dim),dtype=np.float32)

            for i in np.arange(len(X_dev)):
                if pc_sentiment[i] == y_dev[i][0]:
                    correct += 1

        accuracy = correct / len(X_dev)

        print(accuracy)

    @staticmethod
    def train_1layer_glove_wordembedding(hidden_dim, modelfile):
        train = {}
        test = {}
        dev = {}

        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="train",
                                                                             representation="glove.840B.300d")
        embedded_dev, dev_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="dev",
                                                                         representation="glove.840B.300d")
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="test",
                                                                           representation="glove.840B.300d")

        binary_embedded_train = []
        binary_train_labels = []
        for i in np.arange(len(embedded_train)):
            if np.argmax(train_labels[i]) != 1:
                binary_embedded_train.append(embedded_train[i])
                binary_train_labels.append(np.eye(2)[np.argmax(train_labels[i]) // 2])

        binary_embedded_test = []
        binary_test_labels = []
        for i in np.arange(len(embedded_test)):
            if np.argmax(test_labels[i]) != 1:
                binary_embedded_test.append(embedded_test[i])
                binary_test_labels.append(np.eye(2)[np.argmax(test_labels[i]) // 2])

        # train_labels = [np.asarray([(np.argmax(tl) + 1.00) / 3.0]) for tl in train_labels]
        # dev_labels = [np.asarray([(np.argmax(dl) + 1.00) / 3.0]) for dl in dev_labels]
        # embedded_train, train_labels, word_to_index, index_to_word, labels_count = DataPrep.load_one_hot_sentiment_data("../data/sentiment/trainsentence_and_label_binary.txt")
        # embedded_dev, dev_labels= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../data/sentiment/devsentence_and_label_binary.txt",word_to_index,index_to_word,labels_count)
        # self.test["sentences"], self.test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/testsentence_and_label_binary.txt",self.word_to_index, self.index_to_word,self.labels_count)


        flstm = AttendedLSTM(input_dim=len(embedded_train[0][0]), output_dim=2, number_of_layers=1,
                                   hidden_dims=[hidden_dim], dropout_p=0.25, learning_rate=0.01)
        flstm.build_model()

        # train_labels[train_labels == 0] = -1
        # dev_labels[dev_labels == 0] = -1
        flstm.train(binary_embedded_train, binary_train_labels, binary_embedded_test, binary_test_labels)
        flstm.save_model(modelfile)





if __name__ == '__main__':
    #AttendedLSTM.train_finegrained_glove_wordembedding(300, "finetest_model.txt")
    AttendedLSTM.train_1layer_glove_wordembedding(300, "test_model_attended_300.txt")
