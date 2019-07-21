import tensorflow as tf
import numpy as np
import os
import time

class NetKerasModel:

    def __init__(self, setDim, chkpointDir, logDir, batchSize=64, numEpochs=1000, monitor='loss', patience=5):
        self.batch_size = batchSize
        self.num_epochs= numEpochs
        self.set_dim = setDim
        self.monitor = monitor
        self.patience = patience
        self.epoch_size = np.ceil(setDim/self.batch_size)
        self.stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=1)
        self.chkpoints = tf.keras.callbacks.ModelCheckpoint(chkpointDir+"/best_model.h5py", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=None, embeddings_freq=None)
    

    def denseNetKerasWithMean(self, meanInput,  hidden= [], learningRate = 0.005, momentum = 0.9):
        
        def meanBiasInit(shape, dtype, partition_info, name=None):
            return tf.keras.backend.variable(meanInput, name=name)

        def mean_error(self):
            def mean_error_metric(y_true, y_pred):
                dim = tf.constant(self.set_dim, dtype=tf.float32)
                real = tf.math.scalar_mul(dim, y_true)
                pred = tf.math.scalar_mul(dim, y_pred)
                mean_err = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(real,pred)))
                return mean_err

            return mean_error_metric

        self.learningRate = learningRate
        self.momentum = momentum
        self.gradient = 'sgdm'

        model = tf.keras.models.Sequential()
        i = 0

        model.add(tf.keras.layers.InputLayer(input_shape = (64,)))
        #model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dense(64, activation='linear',kernel_initializer = tf.keras.initializers.Identity(), bias_initializer= tf.keras.initializers.Constant(-meanInput), trainable=True))
        
        #model.add(tf.keras.layers.Reshape(target_shape = (1,64,1)))
        #model.add(tf.keras.layers.Reshape(target_shape=(64,)))
        if(len(hidden)):
            for units in hidden:
                model.add(tf.keras.layers.Dense(units,activation=None, kernel_initializer='normal',  kernel_regularizer=tf.keras.regularizers.l2(1.0000e-04)))
                model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.add(tf.keras.layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1.0000e-04)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.summary()
        model.compile(
            loss='mse', 
            optimizer=tf.keras.optimizers.SGD( 
                lr=learningRate,
                momentum=0.9), 
            metrics=['mse','mae', mean_error(self)])
        self.model = model

        return model

    def denseNetKeras(self, hidden= [], learningRate = 0.005, momentum = 0.9):
        
        def mean_error(self):
            def mean_error_metric(y_true, y_pred):
                dim = tf.constant(self.set_dim, dtype=tf.float32)
                real = tf.math.scalar_mul(dim, y_true)
                pred = tf.math.scalar_mul(dim, y_pred)
                mean_err = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(real,pred)))
                return mean_err

            return mean_error_metric
        
        def max_error(self):
            def max_error_metric(y_true, y_pred):
                dim = tf.constant(self.set_dim, dtype=tf.float32)
                #y_pred = tf.nn.relu(y_pred)
                real = tf.math.ceil(tf.math.scalar_mul(dim, y_true))
                pred = tf.math.ceil(tf.math.scalar_mul(dim, y_pred))
                pred = tf.nn.relu(pred)
                max_err = tf.math.reduce_max(tf.math.abs(tf.math.subtract(real,pred)))
                return max_err

            return max_error_metric
        
        def accuracy(self):
            def accuracy_metric(y_true, y_pred):
                dim = tf.constant(self.set_dim, dtype=tf.float32)
                real = tf.math.scalar_mul(dim, y_true)
                pred = tf.math.scalar_mul(dim, y_pred)
                correct = tf.equal(real, pred)
                accuracy = tf.math.reduce_mean(tf.cast(correct, dtype=tf.float32))
                return accuracy

            return accuracy_metric

        self.learningRate = learningRate
        self.momentum = momentum
        self.gradient = 'sgdm'

        model = tf.keras.models.Sequential()
        i = 0
        model.add(tf.keras.layers.InputLayer(input_shape = (64,)))
        #model.add(tf.keras.layers.InputLayer(input_shape = (1,64,1)))
        #model.add(tf.keras.layers.Reshape(target_shape=(64,)))
        #model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
        if(len(hidden)):
            for units in hidden:
                model.add(tf.keras.layers.Dense(units, activation='tanh',kernel_initializer='normal',  kernel_regularizer=tf.keras.regularizers.l2(1.0000e-04)))
                model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.add(tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1.0000e-04)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.summary()
        model.compile(
            loss='mse', 
            optimizer=tf.keras.optimizers.SGD( 
                lr=learningRate,
                momentum=0.9), 
            metrics=['mse','mae', mean_error(self), max_error(self), accuracy(self)])
        self.model = model
        
        return model

    def train(self, features, labels):
        start = time.time()
        self.hystory = self.model.fit(features, labels, epochs=self.num_epochs, batch_size=self.batch_size,  verbose=0, validation_split=0.0, callbacks=[self.stopping, self.chkpoints, self.tensorboard])
        elapsed = (time.time() - start)

        return self.hystory, elapsed
    
    def evaluate(self, features, labels):
        start = time.time()
        self.evaluation = self.model.evaluate(features, labels)
        elapsed = (time.time() - start)

        return self.evaluation, elapsed
    
    def predict(self, features):
        start = time.time()
        len_feat = len(features)
        predictions_prob = self.model.predict(features, self.batch_size)
        predictions = np.multiply(predictions_prob,predictions_prob>0,predictions_prob)
        self.predictions = np.ceil(np.multiply(predictions,self.set_dim))
        elapsed = (time.time() - start)

        return self.predictions, elapsed, predictions_prob

    def setWeigths(self, wDir):
        print("SETTO I PESI")
        if(not self.model):
           raise Exception("Model not Initialized")
        else:
            #print("SONO NELL'ELSE")
            self.model.load_weights(wDir)
            #layer = self.model.layers[0]
            #weights = layer.get_weights()[0]
            #print(weights)

    def setEarlyStopping(self):
        self.stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
    
    def getEarlyStopping(self):
        return self.stopping 

    def getChkpoints(self):
        return self.chkpoints
    
    def getMomentum(self):
        return self.momentum
    
    def getLearningRate(self):
        return self.learningRate

    def getGradient(self):
        return self.gradient
    
    def getMetricMonitor(self):
        return self.monitor
    
    def getStopPatience(self):
        return self.patience

    def getModel(self):
        return self.model

    def getPredictions(self):
        return self.predictions
