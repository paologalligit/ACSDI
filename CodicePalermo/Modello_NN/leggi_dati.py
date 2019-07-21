import pickle

# with open("weights/weights_file1", "rb") as f:
#     ls = pickle.load(f)

with open("weights/weights_file1", "rb") as f:
    ls = pickle.load(f)

# with open("pruned_sizes", "rb") as f:
#     ls = pickle.load(f)
val = ls[2]
tup = val[0]
print(len(tup[1]))
