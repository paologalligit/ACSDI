import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py 
from math import floor

from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from sklearn.metrics import r2_score

RANDOM_SEED = 42
N_CLASSES = 1


start_file = 1
end_file = 9

inner_results = {}

normal_upd = NN.NN.update_layers
normal_mom = NN.NN.updateMomentum

for i in range(start_file, end_file+1):

    with h5py.File('Resource/mat_file/file'+str(i)+'_bin64.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.uint8)
        bin_data = np.flip(bin_data,axis=1)
        dim_set = len(bin_data)

    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
    labels = labels/len(bin_data)
    labels = np.reshape(labels, (-1, 1))

    p = np.random.RandomState(seed=42).permutation(dim_set)

    bin_data_perm = bin_data[p]
    labels_perm = labels[p]

#####################################################################################################################

    pruned_w = []
    
    with open('weights/weights_file{}'.format(i), 'rb') as f:
        w_all = pickle.load(f)

    w = np.copy(w_all[2])

    print('FILE ', i)
    nn = NN.NN(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=0.003, mu=0.9, minibatch=64, disableLog=True)
    nn.addLayers([256], ['leakyrelu','tanh'], w)

    for p in range(10,100,10):
        print('Pruning {}%'.format(p))
        w1=np.copy(w)
        pruning.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=2, num_epochs=20000)
        max_err = 0
        mean_err = 0
        abs_err = 0
        for j in range(dim_set):
            pr = floor(nn.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val
            
            abs_err += abs(labels[j] - nn.predict(bin_data[j])[0])

        pruned_w.append("1 hidden, {}% pruning --> file {}: max error = {} -- mean error = {} -- abs error = {}\n".format(p, i, max_err, mean_err/(len(bin_data)), abs_err / dim_set * 100))

    print('FINITO PRUNING 1 STRATO')
    inner_results['nn_1'] = pruned_w
################################################################################################

    pruned_w = []

    w = np.copy(w_all[3])

    nn = NN.NN(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=0.002, mu=0.9, minibatch=64, disableLog=True)
    nn.addLayers([256, 256], ['leakyrelu','leakyrelu','tanh'], w)
    
    for p in range(10,100,10):
        print('Pruning {}%'.format(p))
        w1=np.copy(w)
        pruning.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=2, num_epochs=20000)
        max_err = 0
        mean_err = 0
        abs_err = 0
        for j in range(dim_set):
            pr = floor(nn.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val
            abs_err += abs(labels[j] - nn.predict(bin_data[j])[0])

        pruned_w.append("2 hidden, {}% pruning --> file {}: max error = {} -- mean error = {} -- abs error = {}\n".format(p, i, max_err, mean_err/(len(bin_data)), abs_err / dim_set * 100))


    inner_results['nn_2'] = pruned_w

    print('FINITO PRUNING 2 STRATI')

    with open('pruning/pruned_results_file{}'.format(i), 'wb') as f:
        pickle.dump(inner_results, f)
   

