import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys

from NN_pr import NN
from NN_pr import WS_module as ws

N_CLASSES = 71

with open('dataset_full_labels', 'rb') as file:
    tools = pickle.load(file)
    
X_scaled = tools['X_scaled']
y_scaled = tools['y_scaled']
X_test_scaled = tools['X_test_scaled']
y_test_scaled = tools['y_test_scaled']
y_test = tools['y_test_full']

print("------------------------no dropout----------------------") 
# 8*256=2048 256*128=32768 128*64=8192
# 71*10=710 10*1=10

std_mom = NN.NN.updateMomentum
std_layers = NN.NN.update_layers

# for n in [[i] for i in range(80, 140, 10)]: #8*300=2400  300*100=3000 100*1=100
for n in [[40], [60], [120], [150], [200]]:
    nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.00007, mu=.99, minibatch=100)
    nn.addLayers(n, ['relu', 'linear'])
    nn.train(0, num_epochs=1000, X_test=X_test_scaled, y_test=y_test_scaled)
    w = (nn.getWeigth())

    size1 = N_CLASSES * n[0]
    size2 = n[0]
    for c in [[int(size1 * .75), int(size2 * .75)], [int(size1 * .5), int(size2 * .5)], [int(size1 * .35), int(size2 * .35)], [int(size1 * .2), int(size2 * .2)]]:
        print("cluster="+str(c))
        w1=np.copy(w)
        ws.set_ws(nn, c, w1)
        nn.train(0, num_epochs=300, X_test=X_test_scaled, y_test=y_test_scaled)
    print('#'*150)

    NN.NN.updateMomentum = std_mom
    NN.NN.update_layers = std_layers

# # print("------------------------dropout=0.75----------------------")        
# # for n in [[i] for i in range(30, 140, 10)]: #28*28*300=235200  300*100=3000 100*10=1000
# #     nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.00007, mu=.99, minibatch=100, dropout=0.75)
# #     nn.addLayers(n, ['relu'])
# #     nn.set_output_id_fun()
# #     nn.train(0, num_epochs=400, X_test=X_test_scaled, y_test=y_test_scaled)
# #     w = (nn.getWeigth())

# #     nn.change_mode()
# #     nn.add_momentum('ws', ws.ws_updateMomentum)
# #     nn.add_layers('ws', ws.ws_update_layers)

# #     for c in [[int(n[0] * .75), int(n[0] * .5)], [int(n[0] * .5), int(n[0] * .35)], [int(n[0] * .35), int(n[0] * .2)], [int(n[0] * .2), int(n[0] * .1)]]:
# #         print("cluster="+str(c))
# #         w1=np.copy(w)
# #         ws.set_ws(nn, c, w1)
# #         nn.train(0, num_epochs=200, X_test=X_test_scaled, y_test=y_test_scaled)
# #     print('#'*150)
        
# # print("------------------------dropout=0.5----------------------")      
# # for n in [[i] for i in range(30, 140, 10)]: #28*28*300=235200  300*100=3000 100*10=1000
# #     nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.00007, mu=.99, minibatch=100, dropout=0.5)
# #     nn.addLayers(n, ['relu'])
# #     nn.set_output_id_fun()
# #     nn.train(0, num_epochs=400, X_test=X_test_scaled, y_test=y_test_scaled)
# #     w = (nn.getWeigth())

# #     nn.change_mode()
# #     nn.add_momentum('ws', ws.ws_updateMomentum)
# #     nn.add_layers('ws', ws.ws_update_layers)

# #     for c in [[int(n[0] * .75), int(n[0] * .5)], [int(n[0] * .5), int(n[0] * .35)], [int(n[0] * .35), int(n[0] * .2)], [int(n[0] * .2), int(n[0] * .1)]]:
# #         print("cluster="+str(c))
# #         w1=np.copy(w)
# #         ws.set_ws(nn, c, w1)
# #         nn.train(0, num_epochs=200, X_test=X_test_scaled, y_test=y_test_scaled)
# #     print('#'*150)

# # '''
# # print("train: "+str(tr))
# # print("test: "+str(te))
# # print("space: "+str(s))

# # fig = plt.figure(figsize=(16,10))
# # ax=fig.add_(1,1,1)
# # ax.plot([0,1],tr[0:1],'.-')

# # for i in range(3):
# #     ax.plot([0,10,20,30],tr[0+i*4,(i+1)*4],'.-')

# # ax.legend("50 n")
# # plt.grid()
# # plt.plot()


# # fig.plt.figure(figsize=(16,10))
# # ax=fig.add_sublot(1,1,1)
# # for i in range(3):
# #     ax.plot([0,10,20,30],te[0+i*4,(i+1)*4],'.-')
# # plt.grid()
# # plt.plot()

# # '''
