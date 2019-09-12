import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys

from NN_pr import NN
from NN_pr import pruning_module as pr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

with open('dataset_full_labels', 'rb') as file:
    tools = pickle.load(file)
    
X_scaled = tools['X_scaled']
print(X_scaled.shape)
y_scaled = tools['y_scaled']
print(y_scaled.shape)
X_test_scaled = tools['X_test_scaled']
print(X_test_scaled.shape)
y_test_scaled = tools['y_test_scaled']
print(y_test_scaled.shape)
y_test = tools['y_test_full']

nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.0007, mu=.99, minibatch=100)#, dropout=0.75)
nn.addLayers([50, 70], ['relu', 'relu', 'linear'])
nn.train(0, num_epochs=1000, X_test=X_test_inner, y_test=y_test_inner)

res = nn.get_res(X_test_inner, y_test_inner)
print('the res: ', res)

# outer_kfold = KFold(n_splits=3, random_state=42, shuffle=False)
# inner_kfold = KFold(n_splits=5, random_state=19, shuffle=False)

# total_results = []

# for fold, (train_index_outer, test_index_outer) in enumerate(outer_kfold.split(X_scaled)):
    
#     X_train_outer, X_test_outer = X_scaled[train_index_outer], X_scaled[test_index_outer]
#     y_train_outer, y_test_outer = y_scaled[train_index_outer], y_scaled[test_index_outer]
    
#     inner_mean_scores = []
#     max_score = -1
#     optimal_neurons = -1

#     neurons = [i for i in range(20, 110, 10)]
    
#     for neuron1 in neurons:
#         for neuron2 in neurons:
#             print('NEURONS {}, {}'.format(neuron1, neuron2))

#             inner_scores = []
            
#             for train_index_inner, test_index_inner in inner_kfold.split(X_train_outer):
#                 X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
#                 y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

#                 nn = NN.NN(training=[X_train_inner, y_train_inner], testing=[X_test_inner, y_test_inner], lr=0.0007, mu=.99, minibatch=100)#, dropout=0.75)
#                 nn.addLayers([neuron1, neuron2], ['relu', 'relu', 'linear'])
#                 nn.train(0, num_epochs=1000, X_test=X_test_inner, y_test=y_test_inner)

#                 res = nn.get_res(X_test_inner, y_test_inner)
#                 inner_scores.append(res['R2'])

#             current_score = np.mean(inner_scores)

#             inner_mean_scores.append({'n_{}_{}'.format(neuron1, neuron2): inner_scores})

#             if max_score < current_score:
#                 max_score = current_score
#                 optimal_neurons = (neuron1, neuron2)
        
#     print('FOLD {} finito'.format(fold))
#     total_results.append({'fold_{}'.format(fold): (max_score, optimal_neurons)})

# print(total_results)

# with open('model_selection_200N_doppio_hiddenLayer', 'wb') as f:
#     pickle.dump(total_results, f)
