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
from LSTMAttentionModel.src.JointAttendedLSTMLayer import *

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
        self.layers = layers




        self.averaged_output = T.dot(self.layers[1].output.T, self.layers[0].output) / T.sum(self.layers[1].output)

        self.predict = theano.function([self.layers[0].input], self.averaged_output)
        self.get_embeding = theano.function([self.layers[0].input], self.layers[0].hidden_state_2)
        self.get_attention = theano.function([self.layers[0].input], self.layers[1].output)



    def build_model(self):
        x = T.matrix('x').astype(theano.config.floatX)
        next_x = T.matrix('n_x').astype(theano.config.floatX)
        y = T.imatrix('y')
        dists = T.ivector('d')
        params = []


        self.layers[0] = JointAttendedLSTMLayer(input=x,
                                           input_dim=self.input_dim,
                                           output_dim=self.hidden_dims[0],
                                           outer_output_dim=self.output_dim,
                                           random_state=self.random_state, layer_id="_0")


        """self.layers[1] = AttendedLSTM1Layer(input=x,
                                           input_dim=self.input_dim,
                                           output_dim=10,
                                           outer_output_dim=1,
                                           random_state=self.random_state, layer_id="_1")

        """


        attention = T.nnet.softmax(self.layers[0].attention.T)[0]
        self.averaged_output = T.dot(attention, self.layers[0].output)
        #T.mean(self.layers[0].output,axis=0)


        output = self.averaged_output #self.layers[0].output[-1] # self.averaged_output #T.nnet.softmax(self.O.dot(self.averaged_output))[0]




        params += self.layers[0].params
        params += self.layers[0].output_params

        #params += self.layers[1].params
        #params += self.layers[1].output_params

        L2 = np.sqrt(sum([ T.sum(param ** 2) for param in params]))

        #cost = T.sum((T.sum(T.nnet.binary_crossentropy((self.layers[0].output + 0.0000001), y),
        #                    axis=1) / dists) )  # T.erf(error1) # T.erf(error1) + + L1 + L2

        cost = T.sum(T.nnet.binary_crossentropy((output+0.000000001),y[-1])) + 0.0001*L2

        updates = LearningAlgorithms.adam(
            cost, params, lr=self.learning_rate
        )

        self.sgd_step = theano.function([x, y], cost, updates=updates)
        self.predict = theano.function([x], output)

        self.test_model = theano.function([x, y], cost)
        self.get_visualization_values = theano.function([x],
                                                        [self.layers[0].output[-1], self.layers[0].hidden_state[-1]])

    def train(self, X_train, y_train, X_dev, y_dev, nepoch=100):
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
                cost = self.sgd_step(np.asarray(X_train[i], dtype=np.float32) * [dropout for i in
                                                                                 np.arange(len(X_train[i]))]
                                     # ,[np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                                     , np.asarray([y_train[i] for k in np.arange(len(X_train[i]))], dtype=np.int32)
                                     # ,np.asarray(next_X,dtype=np.float32)
                                     #,
                                     #np.asarray([pow(c, 2) for c in np.arange(len(X_train[i]), 0, -1)], dtype=np.int32)
                                     )

                """if (iteration % 100 == 0):
                    print("train sample count: " + str(iteration) + " : ")
                    self.test_dev(X_dev, y_dev)
                """
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
        #embedded_dev, dev_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="dev",
        #                                                                 representation="glove.840B.300d")
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
                                   hidden_dims=[hidden_dim], dropout_p=0.2, learning_rate=0.01)
        flstm.build_model()

        # train_labels[train_labels == 0] = -1
        # dev_labels[dev_labels == 0] = -1
        flstm.train(binary_embedded_train, binary_train_labels, binary_embedded_test, binary_test_labels)
        flstm.save_model(modelfile)

    @staticmethod
    def train_finegrained_glove_wordembedding(hidden_dim, modelfile):
        train = {}
        test = {}
        dev = {}

        """ index_to_word = ["UNK"]
        train["sentences"], train["sentiments"], word_to_index, index_to_word, labels_count = DataPrep.load_sentiment_data("../data/sentiment/trainsentence_and_label.txt",index_to_word)
        test["sentences"], test["sentiments"] , word_to_index, index_to_word, lc= DataPrep.load_sentiment_data("../data/sentiment/testsentence_and_label.txt",index_to_word,labels_count)
        dev["sentences"], dev["sentiments"] , word_to_index, index_to_word, lc= DataPrep.load_sentiment_data("../data/sentiment/devsentence_and_label.txt",index_to_word,labels_count)

        vocab_representation = WordEmbeddingLayer()
        vocab_representation.load_embeddings_from_glove_file(filename="../data/glove.840B.300d.txt",filter=index_to_word)
        vocab_representation.save_embedding("../data/finefiltered_glove.840B.300d")
        vocab_representation.load_filtered_embedding("../data/finefiltered_glove.840B.300d")
        vocab_representation.embed_and_save(sentences=train["sentences"],labels=train["sentiments"],path="../data/",name="finetrain",representation="glove.840B.300d")
        vocab_representation.embed_and_save(sentences=dev["sentences"],labels=dev["sentiments"],path="../data/",name="finedev",representation="glove.840B.300d")
        vocab_representation.embed_and_save(sentences=test["sentences"],labels=test["sentiments"],path="../data/",name="finetest",representation="glove.840B.300d")
        """

        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="finetrain",
                                                                             representation="glove.840B.300d")
        embedded_dev, dev_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="finedev",
                                                                         representation="glove.840B.300d")
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="finetest",
                                                                           representation="glove.840B.300d")



        # train_labels = [np.asarray([(np.argmax(tl) + 1.00) / 3.0]) for tl in train_labels]
        # dev_labels = [np.asarray([(np.argmax(dl) + 1.00) / 3.0]) for dl in dev_labels]
        # embedded_train, train_labels, word_to_index, index_to_word, labels_count = DataPrep.load_one_hot_sentiment_data("../data/sentiment/trainsentence_and_label_binary.txt")
        # embedded_dev, dev_labels= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../data/sentiment/devsentence_and_label_binary.txt",word_to_index,index_to_word,labels_count)
        # self.test["sentences"], self.test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/testsentence_and_label_binary.txt",self.word_to_index, self.index_to_word,self.labels_count)


        flstm = AttendedLSTM(input_dim=len(embedded_train[0][0]), output_dim=5, number_of_layers=1,
                             hidden_dims=[hidden_dim], dropout_p=0.25, learning_rate=0.01)
        flstm.build_model()

        # train_labels[train_labels == 0] = -1
        # dev_labels[dev_labels == 0] = -1
        flstm.train(embedded_train, train_labels, embedded_test, test_labels)
        flstm.save_model(modelfile)


    def save_model(self, modelfile):
        with open(modelfile, "wb") as f:
            cPickle.dump(self.layers, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open("params_" + modelfile, "wb") as f:
            for layer_key in self.layers.keys():
                cPickle.dump(self.layers[layer_key].params, f, protocol=cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(modelfile):
        layers = {}
        with open(modelfile, "rb") as f:
            layers = cPickle.load(f)
        with open("params_" + modelfile, "rb") as f:
            for layer_key in layers.keys():
                layers[layer_key].params = cPickle.load(f)

        n_of_layers = len(layers.keys())

        flstm = FullyConnectedLSTM(input_dim=layers[0].input_dim, output_dim=layers[n_of_layers - 1].output_dim,
                                   number_of_layers=n_of_layers, hidden_dims=[layers[0].output_dim])
        flstm.build_loaded_model(layers)
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="test",
                                                                           representation="glove.840B.300d")
        test_labels = [np.asarray([(np.argmax(tl) + 1.00) / 3.0]) for tl in test_labels]
        print("Accuracy on test: ")
        # flstm.test_dev(embedded_test,test_labels)
        print("Accuracy on train: ")
        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/", name="train",
                                                                             representation="glove.840B.300d")
        train_labels = [np.asarray([(np.argmax(tl) + 1.00) / 3.0]) for tl in train_labels]
        # flstm.test_dev(embedded_train, train_labels)
        return flstm

    @staticmethod
    def lstm_forward_pass(embeddedSent, flstm):
        outputs, i_gates, f_gates, o_gates = flstm.forward(embeddedSent)

    @staticmethod
    def show_sentiment_path(sentence, vocab_representation, flstm):
        tokens = sentence.split()
        embedded = vocab_representation.embed([tokens])[0]

        predictions = []
        labels = []
        gates = []
        for i in np.arange(0, len(embedded)):
            labels.append(tokens[i])
            predictions.append(flstm.predict(np.asarray(embedded[0:i + 1], dtype=np.float32)).tolist())
            gates = flstm.get_gates(np.asarray(embedded[0:i + 1], dtype=np.float32))

        vis_data = predictions

        fig = plt.figure()
        ax = {}
        ax[0] = fig.add_subplot(111, projection='3d')

        fig2 = plt.figure()

        for i in np.arange(0, len(gates)):
            for k in np.arange(0, len(embedded)):
                ax[1 + (i * len(tokens)) + k] = fig2.add_subplot(4, len(tokens), (i * len(tokens)) + k + 1)
                ax[1 + (i * len(tokens)) + k].bar(np.arange(len(gates[i][k])), gates[i][k], 0.1)
                ax[1 + (i * len(tokens)) + k].set_ylim(0, 1.0)
                if (i == 0):
                    ax[1 + (i * len(tokens)) + k].set_title(tokens[k])

        vis_x = [x[0] for x in vis_data]
        vis_y = [x[1] for x in vis_data]
        vis_z = [x[2] for x in vis_data]

        ax[0].plot_wireframe(vis_x, vis_y, vis_z, linestyle='-')
        ax[0].scatter(vis_x, vis_y, vis_z, marker='o', depthshade=True)
        for label, x, y, z in zip(labels, vis_x, vis_y, vis_z):
            ax[0].text(x, y, z, label)

        ax[0].set_xlim3d(0, 1)
        ax[0].set_ylim3d(0, 1)
        ax[0].set_zlim3d(0, 1)

        ax[0].set_xlabel('Negative')
        ax[0].set_ylabel('Neutral')
        ax[0].set_zlabel('Positive')
        plt.show()

    @staticmethod
    def show_sentiment_path_1D(sentences, vocab_representation, flstm):

        fig1 = plt.figure()
        fig1.suptitle("sentiment")
        fig2 = plt.figure()
        fig2.suptitle("output gate")
        fig3 = plt.figure()
        fig3.suptitle("forget gate")
        fig4 = plt.figure()
        fig4.suptitle("input gate")

        ax = {}
        ax[0] = fig1.add_subplot(111)
        ax["output_gate"] = fig2.add_subplot(111)
        ax["forget_gate"] = fig3.add_subplot(111)
        ax["input_gate"] = fig4.add_subplot(111)
        sentence_embedings = []
        sentence_sentence = []
        vis_predictions = []
        get_visualization_values = theano.function([flstm.layers[0].input],
                                                   [flstm.layers[0].output[-1], flstm.layers[0].hidden_state[-1]])
        for i in np.arange(0, 2):  # len(sentences)):
            sentence = sentences[i]
            tokens = sentence.split()
            embedded = vocab_representation.embed([tokens])[0]

            predictions = []
            labels = []
            input_gates = []
            output_gates = []
            forget_gates = []

            for i in np.arange(0, len(embedded)):
                labels.append(tokens[i])

                predictions.append(flstm.predict(np.asarray(embedded[0:i + 1], dtype=np.float32)).tolist())
                gates = flstm.get_gates(np.asarray(embedded[0:i + 1], dtype=np.float32))
                input_gates.append(np.dot(flstm.layers[0].output_params[0].get_value(), gates[1][-1].transpose()))
                forget_gates.append(np.dot(flstm.layers[0].output_params[0].get_value(), gates[2][-1].transpose()))
                output_gates.append(np.dot(flstm.layers[0].output_params[0].get_value(), gates[3][-1].transpose()))

            vis_data = predictions

            pred, embding = get_visualization_values(np.asarray(embedded[0:len(embedded)], dtype=np.float32))

            sentence_embedings.append(embding)
            sentence_sentence.append(' '.join(map(str, labels)))
            vis_predictions.append((np.floor(pred[-1] * 3.0) + 1.00) / 3.00)

            vis_x = [i for i in np.arange(len(vis_data))]
            vis_y = [p[0] for p in vis_data]

            ax[0].plot(vis_x, vis_y, linestyle='-')
            ax[0].scatter(vis_x, vis_y, marker='o')
            for label, x, y in zip(labels, vis_x, vis_y):
                ax[0].text(x, y, label)

            vis_data2 = output_gates

            vis_x2 = [i for i in np.arange(len(vis_data2))]
            vis_y2 = [p[0] for p in vis_data2]
            ax["output_gate"].plot(vis_x2, vis_y2, linestyle='-')
            ax["output_gate"].scatter(vis_x2, vis_y2, marker='o')
            for label, x, y in zip(labels, vis_x2, vis_y2):
                ax["output_gate"].text(x, y, label)

            vis_data2 = forget_gates

            vis_x2 = [i for i in np.arange(len(vis_data2))]
            vis_y2 = [p[0] for p in vis_data2]
            ax["forget_gate"].plot(vis_x2, vis_y2, linestyle='-')
            ax["forget_gate"].scatter(vis_x2, vis_y2, marker='o')
            for label, x, y in zip(labels, vis_x2, vis_y2):
                ax["forget_gate"].text(x, y, label)

            vis_data2 = input_gates

            vis_x2 = [i for i in np.arange(len(vis_data2))]
            vis_y2 = [p[0] for p in vis_data2]
            ax["input_gate"].plot(vis_x2, vis_y2, linestyle='-')
            ax["input_gate"].scatter(vis_x2, vis_y2, marker='o')
            for label, x, y in zip(labels, vis_x2, vis_y2):
                ax["input_gate"].text(x, y, label)

        plt.show()

        x = np.asarray(sentence_embedings)
        n_samples, n_features = x.shape

        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(x)

        FullyConnectedLSTM.plot_embedding(X_tsne, np.asarray(vis_predictions), sentence_sentence,
                                          "t-SNE embedding of the embedded digits")

        plt.show()

    @staticmethod
    def plot_embedding(features, classes, labels, title=None):
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        features = (features - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(features.shape[0]):
            plt.text(features[i, 0], features[i, 1], str(labels[i]),
                     color=plt.cm.Set1(float(classes[i]) / 10),
                     fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(features.shape[0]):
                dist = np.sum((features[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [features[i]]]
                """imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)"""
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    @staticmethod
    def analyse():
        flstm = FullyConnectedLSTM.load_model("test_model_diffdim.txt")

        vocab_representation = WordEmbeddingLayer()
        vocab_representation.load_filtered_embedding("../data/filtered_glove.840B.300d")

        FullyConnectedLSTM.show_sentiment_path_1D(
            ["I thought it should be bad , but it was good !", "I thought it should be good , but it was bad !"
                , "they made a bad movie from a good story .", "they made a good movie from a bad story ."
                , "although he is a good actor, his play is bad !"
                , "although he is a bad actor, his play is good !"
                , "the movie made by a good man is bad ."
                , "the movie made by a bad man is good ."
                , "the actor is good ."
                , "the scenario is good ."
                , "terrible !"
                , "great !"
                , "you are good."
                , "the movie is bad !"
                , "the actor is bad !"
                , "the scenario is bad !"
                , "he is a bad actor , but his play is good !"
                , "the bad man made  a good movie ."
                , "the ugly actor played well !"
                , "the movie made by a good man is bad ."
                , "the movie made by a bad man is good ."
                , "the actor is bad, but he played good !"
                , "he played good !"
                , "he played bad !"
                , "they made a bad movie from a good story !"
                , "although he is a bad actor, he played good !"
                , "the bad movie is made by me ."
                , "the bad movie is made by me ."
                , "the bad movie is made by a good man ."
                , "the movie is made  by a good man ."
                , "the movie made  by a good man is bad ."
                , "the movie made  by a bad man is good ."
                , "the bad man made  a good movie ."
                , "the good man made  a bad movie ."
                , "the good man is bad !"
                , "the actor is bad !"
                , "the actor played bad !"
                , "the good actor played so bad !"
                , "I thought it should be bad but it was good !"
                , "I thought it should be bad, but it was good !"
                , "the actor is normally bad, but he played good !"
                , "the actor is bad, but he played good !"
                , "the actor is normally bad !"
                , "the actor is bad !"
                , "he played good !"
                , "his play is good !"
                , "he is a bad actor, he played good !"
                , "he is a bad actor, but he played good !"
                , "he is a bad actor, but his play is good !"
                , "he is a bad actor, but his play is good ! "
                , "although he is a bad actor, he played good !"
                , "although he is a bad actor, his play is good !"
                , "although he is a bad actor, his act is good !"
                , "although he is a bad actor, his play is good !"
                , "they made a bad movie from a good story !"
             ], vocab_representation, flstm)
        embed_sent = vocab_representation.embed(sentences=[["bad", "!"]])[0]
        print("bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["not", "bad", "!"]])[0]
        print("not bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["very", "bad", "!"]])[0]
        print("very bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["it", "is", "not", "good", "!"]])[0]
        print("it is not good! is: " + str(np.argmax(flstm.predict(np.asarray(embed_sent, dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["it", "is", "not", "good", "not", "bad", "!"]])[0]
        print("it is not good not bad! is: " + str(np.argmax(flstm.predict(np.asarray(embed_sent, dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["good", "or", "bad", "!"]])[0]
        print("good or bad! is: " + str(np.argmax(flstm.predict(np.asarray(embed_sent, dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["good", "or", "bad", "!"]])[0]
        print("bad or good! is: " + str(np.argmax(flstm.predict(np.asarray(embed_sent, dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["the"]])[0]
        print("the is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["bad"]])[0]
        print("bad is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["movie"]])[0]
        print("movie is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["is"]])[0]
        print("is is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["made"]])[0]
        print("made is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["by"]])[0]
        print("by is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["me"]])[0]
        print("me is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "bad", "movie", "is", "made", "by", "me", "."]])[0]
        print("the bad movie is made by me. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(
                sentences=[["the", "bad", "movie", "is", "made", "by", "a", "good", "man", "."]])[0]
        print(
            "the bad movie is made by a good man. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(sentences=[["the", "movie", "is", "made", "by", "a", "good", "man", "."]])[0]
        print("the movie is made  by a good man. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(
                sentences=[["the", "movie", "made", "by", "a", "good", "man", "is", "bad", "."]])[0]
        print(
            "the movie made  by a good man is bad. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(
                sentences=[["the", "movie", "made", "by", "a", "bad", "man", "is", "good", "."]])[0]
        print(
            "the movie made  by a bad man is good. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "bad", "man", "made", "a", "good", "movie", "."]])[0]
        print("the bad man made  a good movie. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "good", "man", "made", "a", "bad", "movie", "."]])[0]
        print("the good man made  a bad movie. is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "good", "man", "is", "bad", "!"]])[0]
        print("the good man is bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "actor", "is", "bad", "!"]])[0]
        print("the actor is bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "actor", "played", "bad", "!"]])[0]
        print("the actor played bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "good", "actor", "played", "bad", "!"]])[0]
        print("the good actor played so bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["I", "thought", "it", "should", "be", "bad", "but", "it", "was", "good", "!"]])[0]
        print("I thought it should be bad but it was good ! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["I", "thought", "it", "should", "be", "bad", ",", "but", "it", "was", "good", "!"]])[0]
        print("I thought it should be bad, but it was good ! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["the", "actor", "is", "normally", "bad", ",", "but", "he", "played", "good", "!"]])[0]
        print("the actor is normally bad, but he played good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(
                sentences=[["the", "actor", "is", "bad", ",", "but", "he", "played", "good", "!"]])[
                0]
        print(
            "the actor is bad, but he played good! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "actor", "is", "normally", "bad", "!"]])[0]
        print("the actor is normally bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the", "actor", "is", "bad", "!"]])[0]
        print("the actor is bad! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he", "played", "good", "!"]])[0]
        print("he played good! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["his", "play", "is", "good", "!"]])[0]
        print("his play is good! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = \
            vocab_representation.embed(sentences=[["he", "is", "a", "bad", "actor", "he", "played", "good", "!"]])[0]
        print("he is a bad actor, he played good! is: " + str(flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["he", "is", "a", "bad", "actor", ",", "but", "he", "played", "good", "!"]])[0]
        print("he is a bad actor, but he played good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["he", "is", "a", "bad", "actor", ",", "but", "his", "play", "is", "good", "!"]])[0]
        print("he is a bad actor, but his play is good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["he", "is", "a", "bad", "actor", ",", "but", "his", "play", "is", "good", "!"]])[0]
        print("he is a bad actor, but his play is good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["although", "he", "is", "a", "bad", "actor", ",", "he", "played", "good", "!"]])[0]
        print("although he is a bad actor, he played good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["although", "he", "is", "a", "bad", "actor", ",", "his", "play", "is", "good", "!"]])[0]
        print("although he is a bad actor, his play is good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["although", "he", "is", "a", "bad", "actor", ",", "his", "act", "is", "good", "!"]])[0]
        print("although he is a bad actor, his act is good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["although", "he", "is", "a", "bad", "actor", ",", "his", "play", "is", "good", "!"]])[0]
        print("although he is a bad actor, his play is good! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))

        embed_sent = vocab_representation.embed(
            sentences=[["they", "made", "a", "bad", "movie", "from", "a", "good", "story", "!"]])[0]
        print("they made a bad movie from a good story! is: " + str(
            flstm.predict(np.asarray(embed_sent, dtype=np.float32))))




if __name__ == '__main__':
    #AttendedLSTM.train_finegrained_glove_wordembedding(300, "finetest_model.txt")
    model = "orthogonal_jointattended_2Layer_100-D02.txt"
    print(model)
    AttendedLSTM.train_1layer_glove_wordembedding(100, model)
