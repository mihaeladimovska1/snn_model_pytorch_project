'''So the idea: group testing set by classes; get the avergae of each filter output after a forward pass of a testing image from class c;
remove all filters with low average--> these neurons don't fire on a 7 :)
How is this different than the NISP article? There they remove fitlers which don't disturb the next layer.
Thus, they might remove both filters with low and high average. But in our case, we can be like: well, removing ALL filters with high average
on a 7, makes the network unable to recognize a 7?'''

import os
print(os.environ['XDG_RUNTIME_DIR'])
import pickle
import numpy as np


def load_predefined_kernels(path):
    kernels = []
    kernel = np.empty((3, 3))
    f = open(path, "r")#open("kernels_3x3.csv", "r")
    lines = f.readlines()
    k = 0
    for line in lines:
        elements = line.split('\t')
        kernel[k, 0] = float(elements[0])
        kernel[k, 1] = float(elements[1])
        kernel[k, 2] = float(elements[2])
        k += 1
        if k == 3:
            k = 0
            kernels.append(kernel)
            kernel = np.empty((3, 3))
    return kernels


def get_all_labels(test_loader):
    labels = []
    for data, target in test_loader:
        labels.append(target.numpy().flatten())
    labels = np.asarray(labels)
    return np.hstack(labels)

def get_all_data(test_loader):
    data_all = []
    for data, target in test_loader:
        data_all.append(data)
    #data_all = np.asarray(data_all)
    return np.vstack(data_all)

def get_groups_of_class_indices(true_labels, num_classes):
    groups = [[] for x in range(num_classes)]
    for i in range(num_classes):
        groups[i].append(np.where(np.equal(true_labels, i))[0])
    return groups



def load_learned_weights(path):
    kernels = []
    kernel = np.empty((3, 3))
    f = open(path, "r")  # open("kernels_3x3.csv", "r")
    lines = f.readlines()
    k = 0
    for line in lines:
        elements = line.split('\t')
        kernel[k, 0] = float(elements[0])
        kernel[k, 1] = float(elements[1])
        kernel[k, 2] = float(elements[2])
        k += 1
        if k == 3:
            k = 0
            kernels.append(kernel)
            kernel = np.empty((3, 3))
    return kernels

