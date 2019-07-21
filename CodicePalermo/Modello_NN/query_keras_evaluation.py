import pickle
import os
import h5py
import tensorflow as tf
import numpy as np
from NetKerasModel import NetKerasModel

start_file = 1
end_file = 9
learningRate = 0.5
momentum = 0.9

batch_size = 64
num_epochs = 20000
#perc_init = 10
#perc_step = 10

dirExpName = "KerasExperiment1"
dirUtilsName = 'utils'

for i in range(start_file, end_file+1):

    result = []
    with h5py.File('../mat_file/file'+str(i)+'_bin64.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.bool) # For converting to numpy array
    bin_data = np.flip(bin_data,axis=1)
    dim_set = len(bin_data)
    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
    labels = labels/len(bin_data)
    labels = np.reshape(labels, (-1, 1))

    perc_list = np.linspace(10, 100, num=10, dtype=np.float64)
    result = []

    #model = tf.keras.models.load_model('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer/best_model.h5py')
    #print(model)

    kerasModel1 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer', './Result/file'+str(i)+'/'+dirExpName+'/logs/1-layer', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel1.denseNetKeras(hidden= [], learningRate = learningRate, momentum = momentum)
    #model = model.load_model('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer/best_model.h5py')
    kerasModel1.setWeigths('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/1-layer/best_model.h5py')
    
    kerasModel2 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/2-layers', './Result/file'+str(i)+'/'+dirExpName+'/logs/2-layers', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel2.denseNetKeras(hidden= [256], learningRate = learningRate, momentum = momentum)
    kerasModel2.setWeigths('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/2-layers/best_model.h5py')
    
    kerasModel3 = NetKerasModel(dim_set, './Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/3-layers', './Result/file'+str(i)+'/'+dirExpName+'/logs/3-layers', batchSize=batch_size, numEpochs=num_epochs)
    model = kerasModel3.denseNetKeras(hidden= [256,256], learningRate = learningRate, momentum = momentum)
    kerasModel3.setWeigths('./Result/file'+str(i)+'/'+dirExpName+'/ckpt_models/3-layers/best_model.h5py')
    
    
    for perc in perc_list:
        data_query = bin_data[0:np.floor(perc*dim_set/100).astype(np.int64)]
        labels_query = labels[0:np.floor(perc*dim_set/100).astype(np.int64)]

        result1, predTime1 = kerasModel1.evaluate(data_query, labels_query)
        result2, predTime2 = kerasModel2.evaluate(data_query, labels_query)
        result3, predTime3 = kerasModel3.evaluate(data_query, labels_query)
        
        real = np.multiply(labels_query,dim_set)

        pred1 = kerasModel1.predict(data_query)
        error1 = np.abs(pred1[0] - real)
        maxerr1 = np.max(error1)
        meanerr1 = np.mean(error1)

        pred2 = kerasModel2.predict(data_query)
        error2 = np.abs(pred2[0] - real)
        maxerr2 = np.max(error2)
        meanerr2 = np.mean(error2)

        pred3 = kerasModel3.predict(data_query)
        error3 = np.abs(pred3[0] - real)
        maxerr3 = np.max(error3)
        meanerr3 = np.mean(error3)

        result1[3] = meanerr1
        result2[3] = meanerr2
        result3[3] = meanerr3
        result1[4] = maxerr1
        result2[4] = maxerr2
        result3[4] = maxerr3

        result.append([(result1, predTime1),( result2, predTime2),(result3, predTime3)])
        
    epsilon = [np.ceil(result1[4]), np.ceil(result2[4]), np.ceil(result3[4])]


    with open('./Result/file'+str(i)+'/'+dirExpName+'/query_eval.pickle','wb') as fp:
        pickle.dump(result, fp)

    with open('./Result/file'+str(i)+'/'+dirExpName+'/query_epsilon.pickle','wb') as fp:
        pickle.dump(epsilon, fp)






    '''
    perc_list = np.linspace(10, 100, num=10, dtype=np.float64)

    for perc in perc_list:
        data_query = bin_data[0:np.floor(perc*dim_set/100).astype(np.int64)]
        labels_query = labels[0:np.floor(perc*dim_set/100).astype(np.int64)]

        query_result = []

        for estimator in est_list:
            evalEst, predTime =  estimator.evaluate(data_query, labels_query)
            query_result.append((evalEst, predTime))
        result.append(query_result)

    with open('./Result/file'+str(i)+'/'+dirExpName+'/query_eval.pickle','wb') as fp:
        pickle.dump(result, fp)
    '''

