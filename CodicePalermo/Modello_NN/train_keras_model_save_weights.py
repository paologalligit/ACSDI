import tensorflow as tf
import numpy as np
import h5py 
import time
import os
import pickle
import matplotlib.pyplot as plt
from NetKerasModel import NetKerasModel
from Utils.weights import WeightsUtils

start_file = 2
end_file = 2

dirExpName = "KerasExperiment2"
dirUtilsName = "utils"

part_split = 8

batch_size = 64
num_epochs = 20000
#tf.logging.set_verbosity(tf.logging.INFO)

for i in range(start_file, end_file+1):

    trainingTime = []
    learningRate = 0.1
    momentum = 0.9

    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer')   
    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/2-layers')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/2-layers') 
    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/3-layers')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/3-layers')

    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/logs/1-layer')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/logs/1-layer')   
    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/logs/2-layers')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/logs/2-layers') 
    if(not os.path.isdir('./Result/file'+str(i)+'/'+dirExpName+'/logs/3-layers')):
        os.makedirs('./Result/file'+str(i)+'/'+dirExpName+'/logs/3-layers') 
    
    with h5py.File('Resource/mat_file/file'+str(i)+'_bin64.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.bool) # For converting to numpy array
    bin_data = np.flip(bin_data,axis=1)
    dim_set = len(bin_data)
    print(dim_set)
    print(bin_data)
    

    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
    labels = labels/len(bin_data)
    labels = np.reshape(labels, (-1, 1))
    
    p = np.random.permutation(dim_set)

    bin_dataOld = bin_data
    bin_data = bin_data[p]
    
    labels = labels[p]
    
    #batch_size = dim_set
    param_dict={
	    "setDim" : dim_set,
	    "filename" : 'file'+str(i),
	    "batchSize" : batch_size,
	    "numEpochMax" : num_epochs,
	    "numEpochReal" : [],
        "stopMonitor" : [],
        "stopPatience" : [],
        "learningRate" : learningRate,
        "gradient": [],
        "momentum" : momentum
    }

    with open('./Result/file'+str(i)+'/'+dirExpName+'/params.pickle','wb') as fp:
        pickle.dump(param_dict, fp, pickle.HIGHEST_PROTOCOL)
    #bin_data = 2*bin_data - 1
    #m = np.mean(bin_data, axis=0, dtype=np.float16)
    print(len(bin_data))

    d = {}

    kerasModel1 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer', './Result/file'+str(i)+'/'+dirExpName+'/logs/1-layer', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel1.denseNetKeras( hidden= [], learningRate = learningRate, momentum = momentum)
    history1, tt1 = kerasModel1.train(bin_data, labels)
    param_dict["numEpochReal"].append(len(history1.epoch))
    param_dict["gradient"].append(kerasModel1.getGradient())
    param_dict["stopMonitor"].append(kerasModel1.getMetricMonitor())
    param_dict["stopPatience"].append(kerasModel1.getStopPatience())
    trainingTime.append(tt1)

    model.save_weights('pesi.h5')

    W = WeightsUtils('pesi.h5')
    w = W.get_weights()

    d[1] = w
    
    kerasModel2 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/2-layers', './Result/file'+str(i)+'/'+dirExpName+'/logs/2-layers', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel2.denseNetKeras(hidden= [256], learningRate = learningRate, momentum = momentum)
    history2, tt2 = kerasModel2.train(bin_data, labels)
    param_dict["numEpochReal"].append(len(history2.epoch))
    param_dict["gradient"].append(kerasModel2.getGradient())
    param_dict["stopMonitor"].append(kerasModel2.getMetricMonitor())
    param_dict["stopPatience"].append(kerasModel2.getStopPatience())
    trainingTime.append(tt2)

    model.save_weights('pesi.h5')

    W = WeightsUtils('pesi.h5')
    w = W.get_weights()

    d[2] = w

    kerasModel3 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/3-layers', './Result/file'+str(i)+'/'+dirExpName+'/logs/3-layers', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel3.denseNetKeras(hidden= [256,256], learningRate = learningRate, momentum = momentum)
    history3, tt3 = kerasModel3.train(bin_data, labels)
    param_dict["numEpochReal"].append(len(history3.epoch))
    param_dict["gradient"].append(kerasModel3.getGradient())
    param_dict["stopMonitor"].append(kerasModel3.getMetricMonitor())
    param_dict["stopPatience"].append(kerasModel3.getStopPatience())
    trainingTime.append(tt3)

    model.save_weights('pesi.h5')

    W = WeightsUtils('pesi.h5')
    w = W.get_weights()

    d[3] = w

    with open('weights_file{}'.format(i), 'wb') as f:
        pickle.dump(d, f)

    # print(model.predict(bin_dataOld[0].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[5].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[10].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[35].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[100].reshape((1, -1))) * len(bin_data))

    
    # print(model.predict(bin_dataOld[-1].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[-3].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[-5].reshape((1, -1))) * len(bin_data))
    # print(model.predict(bin_dataOld[-10].reshape((1, -1))) * len(bin_data))
    # x = np.ones((1, 64))
    # print(model.predict(x) * len(bin_data))


    # with open('./Result/file'+str(i)+'/'+dirExpName+'/params.pickle','wb') as fp:
    #     pickle.dump(param_dict, fp, pickle.HIGHEST_PROTOCOL)

    # with open('./Result/file'+str(i)+'/'+dirExpName+'/training_time.pickle','wb') as fp:
    #     pickle.dump(trainingTime, fp, pickle.HIGHEST_PROTOCOL)
    