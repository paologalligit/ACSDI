from __future__ import print_function
import h5py
import numpy as np

class WeightsUtils:

    def __init__(self, source):
        self.source = source

    def extract_weights(self, file, debug):
        f = h5py.File(file)
        d = {}
        try:
            layer_count = 1
            for layer, g in f.items():
                for p_name in g.keys():
                    param = g[p_name]
                    for k_name in param.keys():
                        ls = []
                        for i in param.get(k_name):
                            ls.append(i)
                        if debug: print("      {}/{}: {} x {}".format(p_name, k_name, len(ls), len(ls[0]) if k_name.startswith('kernel') else 1))
                        label = k_name.split(':')[0]
                        d['layer_{}_{}'.format(label, layer_count)] = np.array(ls)
                    layer_count += 1
        finally:
            f.close()

        return d
    
    def get_weights(self, debug=False):
        if isinstance(self.source, str):
            weights_dict = self.extract_weights(self.source, debug)

            w, b = [], []
            for k, v in weights_dict.items():
                if k.startswith('layer_kernel'): 
                    w.append(v)
                    print('w --> ', v.shape)
                else: 
                    b.append(v.reshape((1, -1)))
                    print('b --> ', v.reshape((1, -1)).shape)
                
            return [i for i in zip(w, b)]
        
        else:
            w = self.source.get_weights()
            return [i for i in zip(*[iter(w)] * 2)]