import pickle

with open('weights_file1', 'rb') as f:
    ls = pickle.load(f)

print(ls)