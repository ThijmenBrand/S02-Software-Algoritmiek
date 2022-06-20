from calendar import c
import numpy as np
from collections import defaultdict
import collections
from itertools import islice
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords

class word2vec():
    def __init__(self, epochs):
        self.N = 5
        self.eta = 0.03
        self.epochs = epochs
        self.window = 2

    def generate_training_data(self, corpus):
        corpus = corpus.split('.')
        corpus = [x.split() for x in corpus]
        
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        print(corpus)

        self.v_count = len(word_counts.keys())

        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)

            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def softmax(self, x):
        e_x = np.exp(x -np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    def train(self, training_data):
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.N))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.N, self.v_count))

        for i in range(0, self.epochs):
            self.loss = 0

            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_pass(w_t)

                e = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                self.backprop(e, h, w_t)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print("epoch:", i, "Loss:", self.loss)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def word_sim(self, word, top_n):
        v_w1 = self.word_vec(word)

        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            self.word = self.index_word[i]
            if (word != self.word):
                word_sim[self.word] = theta

        words_sorted = collections.OrderedDict(sorted(word_sim.items(), key=lambda x: x[1], reverse=True))
        return list(islice(words_sorted, top_n))

    def similarity(self, word1, word2):
        v_w1 = self.word_vec(word1)
        v_w2 = self.word_vec(word2)

        return np.dot(v_w1,v_w2)/(np.linalg.norm(v_w1)*np.linalg.norm(v_w2))

    def plot_vectors(self):
        for i, word in enumerate(self.words_list):
            x, y = float(self.w1[i][0]), float(self.w1[i][1])
            plt.scatter(x,y)
            plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()


corpus = """he is a king. she is a queen. he is a man. she is a woman. paris is the capital of france. the capital of france is paris. the capital of italy is rome. rome is the capital of italy. a queen is she. he is a king and a man"""
w2v = word2vec(10_000)
training_data = w2v.generate_training_data(corpus)
w2v.train(training_data)
w2v.plot_vectors()
