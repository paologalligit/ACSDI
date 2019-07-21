import numpy as np
import math
from NN_pr import logger as log
from NN_pr import activation_function as af

from sklearn.metrics import r2_score

N_FEATURES = 71
N_CLASSES = 1


class NN:
    def __init__(self, training, testing, lr, mu, minibatch, dropout=None, disableLog=None, weights=None):
        self.training = training
        self.testing = testing
        self.numEx = len(self.training[0])
        self.numTest = len(self.testing[0])
        self.lr = lr
        self.mu = mu
        self.minibatch = minibatch
        self.p = dropout
        self.disableLog = disableLog
        self.layers = weights

        self.target_train = self.training[1]
        self.target_test = self.testing[1]
        self.mask = 0
        self.epoch = 0
        
        self.layers_shape = []
        self.centers = []
        self.idx_layers = []  
        self.cluster=0      

        
        
        # self.targetForUpd = np.zeros((self.numEx, N_CLASSES), dtype=int)
        # for i in range(self.numEx):
        #     self.targetForUpd[i, training[1][i]] = 1
         
            
    def addLayers(self, neurons, activation_fun):
        self.epoch = 0
        log.logNN.info("neurons= "+str(neurons))
        self.nHidden = len(neurons)
        self.layers = []
        self.v = []
        self.act_fun = []
        for i in range(self.nHidden+1):
            if activation_fun[i] == 'relu':
                self.act_fun.append(lambda x, der: af.ReLU(x, der))
            elif activation_fun[i] == 'sigmoid':
                self.act_fun.append(lambda x, der: af.sigmoid(x, der))
            elif activation_fun[i] == 'linear':
                self.act_fun.append(lambda x, der: af.linear(x, der))
            elif activation_fun[i] == 'tanh':
                self.act_fun.append(lambda x, der: af.tanh(x, der))
            elif activation_fun[i] == 'leakyrelu':
                self.act_fun.append(lambda x, der: af.LReLU(x, der))
        
        for i in range(self.nHidden):    
            n = neurons[i]
            Wh = np.random.randn(N_FEATURES if i == 0 else neurons[i - 1], n) * math.sqrt(2.0 / self.numEx)
            bWh = np.random.randn(1, n) * math.sqrt(2.0 / self.numEx)
            self.layers.append([Wh, bWh])
            self.v.append([0, 0])
        Wo = np.random.randn(neurons[-1], N_CLASSES) * math.sqrt(2.0 / self.numEx)
        bWo = np.random.randn(1, N_CLASSES) * math.sqrt(2.0 / self.numEx)
        self.layers.append([Wo, bWo])
        self.v.append([0, 0])
        

    def predict(self, X):
        outputs = []
        inputLayer = X
        for i in range(self.nHidden):             
            H = self.act_fun[i](np.dot(inputLayer, self.layers[i][0]) + self.layers[i][1], False)
            outputs.append(H)
            inputLayer = H
        outputs.append(self.act_fun[-1]((np.dot(H, self.layers[-1][0]) + self.layers[-1][1]), False))
        return outputs

    def predictHotClass(self, X):
        return np.argmax(self.predict(X)[-1], axis=1).reshape(-1, 1)

    def accuracy(self, X, t):
        lengthX = X.shape[0]
        correct = 0
        predictions = self.predict(X)[-1]

        return np.sum(np.abs(predictions - t), axis=0) / lengthX
    
    def train_acc(self, X, t):
        lengthX = X.shape[0]
        correct = 0
        predictions = self.predict(X)[-1]

        return round(r2_score(t, predictions) * 100, 2)

    def updateMomentum(self, X, t, nEpochs, learningRate, momentumUpdate):
        numBatch = (int)(self.numEx / self.minibatch)
        max_learning_rate = learningRate
        min_learning_rate = 0.0001
        # decay_speed = 100.0
        decay_speed = 0.01
        # lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-self.epoch/decay_speed)
        lr = self.lr * (1 / (1 + decay_speed * self.epoch))
        # lr = learningRate

        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            outputs = self.predict(X[indexLow:indexHigh])
            if self.p != None:
                for i in range(len(outputs) - 1):
                    mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                    outputs[i] *= mask

            y = outputs[-1]
            deltas = []

            # calcolo errore
            deltas.append(self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh])) # * (1 / self.minibatch))

            # backpropagation
            for i in range(self.nHidden):
                deltas.append(
                    np.dot(deltas[i], self.layers[self.nHidden - i][0].T) * self.act_fun[self.nHidden - i - 1](outputs[self.nHidden - i - 1], True))
            
            deltas.reverse()

            deltasUpd = []
            deltasUpd.append([- lr * np.dot(X[indexLow:indexHigh].T, deltas[0]), - lr * np.sum(deltas[0], axis=0, keepdims=True)])
            for i in range(self.nHidden):
                deltasUpd.append(
                    [- lr * np.dot(outputs[i].T, deltas[i + 1]), - lr * np.sum(deltas[i + 1], axis=0, keepdims=True)])

            self.update_layers(deltasUpd, momentumUpdate)

            
    def update_layers(self, deltasUpd, momentumUpdate):
        for i in range(self.nHidden + 1):
                self.layers[i][0] += deltasUpd[i][0] + momentumUpdate * self.v[i][0] 
                self.layers[i][1] += deltasUpd[i][1] + momentumUpdate * self.v[i][1]
        self.v = deltasUpd
        
    def stop_fun(self, t=0, num_epochs=None, acc_epoch=None):
        if t==0:
            #numero epoche
            if num_epochs > self.epoch:
                return True
            else:
                return False
        elif t==1:
            if self.epoch <= 50 or acc_epoch - self.best_acc_test >= 0.01:
                self.best_acc_test=acc_epoch
                return True
            else:
                return False
            
    def print_res(self, X_test, y_test):
        res = self.predict(X_test)[-1]
        return 'res {}, R2 {}%'.format(np.sum(np.abs(res - y_test), axis=0) / len(y_test), round(r2_score(y_test, res) * 100, 2))       

    def get_res(self, X_test, y_test):
        res = self.predict(X_test)[-1]
        # scaler_test = StandardScaler().fit(y_test)
        # reversed_res = scaler_test.inverse_transform(res)
        return {'res': np.sum(np.abs(res - y_test), axis=0) / len(y_test), 'R2': round(r2_score(y_test, res) * 100, 2)}

    def train(self, stop_function, num_epochs=None, X_test=[], y_test=[]):
        train = self.training[0]
        test = self.testing[0]
        
        self.best_acc_test = 0.
        last_acc = 1.
        
        if self.disableLog:
            log.logNN.disabled=True

        log.logNN.info("learning rate=" + str(self.lr) + " momentum update=" + str(self.mu) + " minibatch=" + str(self.minibatch))
        while self.stop_fun(stop_function, num_epochs, last_acc):
            self.updateMomentum(train, self.training[1], num_epochs, self.lr, self.mu)

            if self.epoch % 5 == 0:
                log.logNN.debug("Accuracy - epoch " + str(self.epoch) + ":  Train=" + str(
                    self.accuracy(train, self.target_train)) + "- Test=" + str(self.accuracy(test, self.target_test)))
            last_acc = self.accuracy(test, self.target_test)
            self.epoch += 1

        log.logNN.info("Train acc - epoch: " + str(self.epoch) + ":  " + str(self.accuracy(train, self.target_train)) + ", " + str(self.train_acc(train, self.target_train)) + "%")
        log.logNN.info("Test acc - epoch:" + str(self.accuracy(test, self.target_test)))
        log.logNN.info(self.print_res(X_test, y_test))
        log.logNN.debug("-------------------------------------------------------------------------------")
        return self.accuracy(train, self.target_train), self.accuracy(test, self.target_test)

    def getWeigth(self):
        return self.layers
