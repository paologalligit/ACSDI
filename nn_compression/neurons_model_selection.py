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
print(X_scaled.max())
y_scaled = tools['y_scaled']
X_test_scaled = tools['X_test_scaled']
y_test_scaled = tools['y_test_scaled']
y_test = tools['y_test_full']

outer_kfold = KFold(n_splits=3, random_state=42, shuffle=False)
inner_kfold = KFold(n_splits=5, random_state=19, shuffle=False)

total_results = []

for fold, (train_index_outer, test_index_outer) in enumerate(outer_kfold.split(X_scaled)):
    
    X_train_outer, X_test_outer = X_scaled[train_index_outer], X_scaled[test_index_outer]
    y_train_outer, y_test_outer = y_scaled[train_index_outer], y_scaled[test_index_outer]
    
    inner_mean_scores = []

    neurons = [[i] for i in range(20, 130, 10)]
    
    for neuron in neurons:
        
        inner_scores = []
        
        for train_index_inner, test_index_inner in inner_kfold.split(X_train_outer):
            X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            nn = NN.NN(training=[X_train_inner, y_train_inner], testing=[X_test_inner, y_test_inner], lr=0.0007, mu=.99, minibatch=100)#, dropout=0.75)
            nn.addLayers(neuron, ['relu', 'linear'])
            nn.train(0, num_epochs=1000, X_test=X_test_inner, y_test=y_test_inner)

            res = nn.get_res(X_test_inner, y_test_inner)
            inner_scores.append(
                {'R2': res['R2']}
            )
                    
        inner_mean_scores.append({'n_{}'.format(neuron): inner_scores})
    
    print('FOLD {}'.format(fold))
    total_results.append({'fold_{}'.format(fold): inner_mean_scores})

print(total_results)

with open('model_selection_res_low_lr', 'wb') as f:
    pickle.dump(total_results, f)
