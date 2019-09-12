import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys

from NN_pr import NN
from NN_pr import pruning_module as pr

from sklearn.preprocessing import StandardScaler

with open('dataset_full_labels', 'rb') as file:
    tools = pickle.load(file)
    
X_scaled = tools['X_scaled']
print(X_scaled.max())
y_scaled = tools['y_scaled']
X_test_scaled = tools['X_test_scaled']
y_test_scaled = tools['y_test_scaled']
y_test = tools['y_test_full']

print(X_scaled.shape)
print(y_scaled.shape)
print(X_test_scaled.shape)
print(y_test_scaled.shape)
print(y_test.shape)
print(np.mean(y_test))

def coeff_determination(y_test, y_pred):
    from keras import backend as K
    SS_res =  np.sum(( y_test-y_pred ) ** 2)
    SS_tot = np.sum(( y_test - np.mean(y_test) ) ** 2)
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

tr = []
te=[]
s=[]
normal_upd = NN.NN.update_layers
for n in [[60], [70], [90], [110], [120]]:
    print('############################### NEURONS {} ###############################'.format(n[0]))
    nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.01, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu', 'linear'])
    a,b=nn.train(0, num_epochs=500, X_test=X_test_scaled, y_test=y_test_scaled)

    w = (nn.getWeigth())
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        print('@'*60, '\n')
        flat_list1 = [item for sublist in nn.mask[0] for item in sublist]
        flat_list2 = [item for sublist in nn.mask[1] for item in sublist]
        pruned1, pruned2 = flat_list1.count(False), flat_list2.count(False)
        print('             INPUT/HIDDEN MASK: ', pruned1, round((pruned1 *100) / (71 * n[0]), 4), '%')
        print('             HIDDEN/OUTPUT MASK: ', pruned2, round((pruned2 * 100) / n[0], 4), '%\n')
        print('@'*60, '\n')
        a,b=nn.train(0, num_epochs=50, X_test=X_test_scaled, y_test=y_test_scaled)

    print('############################### END ###############################\n\n')

    # with open('training_res', 'wb') as file:
    #     pickle.dump(res, file)