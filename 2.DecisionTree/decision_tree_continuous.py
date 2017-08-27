#!/usr/bin/python
# coding: utf-8

# dataset:
# https://alliance.seas.upenn.edu/~cis520/wiki/index.php?n=Lectures.DecisionTrees


import numpy as np
import pandas as pd
from pprint import pprint

print("Decision Tree in Python.")


def create_data():
    labeled_data = pd.read_csv("./data.csv", header=None)
    return np.array(labeled_data)


def entropy(y):
    # H = sum(-p * log2(p))
    res = 0.0
    unique, counts = np.unique(y, return_counts=True)
    freqs = counts.astype('float') / len(y)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def partition_discrete(x_i):
    return {str(val): (x_i == val).nonzero()[0] for val in np.unique(x_i)}


def partition_continue(x_i, threshold):
    groups = {}
    for idx, val in enumerate(x_i):
        if val >= threshold:
            key = '>=' + str(threshold)
        else:
            key = '<' + str(threshold)
        groups.setdefault(key, []).append(idx)
    return groups


def info_gain_discrete(x_i, y, type='entropy'):
    # I(y,x)=H(y)−[px=0 H(y|x=0)+px=1 H(y|x=1))]
    res = entropy(y)
    values, counts = np.unique(x_i, return_counts=True)
    freqs = counts.astype('float') / len(x_i)
    for val, p in zip(values, freqs):
        res -= p * entropy(y[x_i == val])
    return res, 0


def info_gain_continue(x_i, y):
    # I(y,x)=H(y)−[px=0 H(y|x=0)+px=1 H(y|x=1))]
    res = entropy(y)
    # sort x_i
    x_i_sorted = np.sort(x_i)
    # unique, counts = np.unique(x_i, return_counts=True)
    min_entropy = np.finfo(np.float64).max
    threshold = 0
    for thred in x_i_sorted[:-1]:
        p = float(len((x_i >= thred).nonzero()[0])) / len(x_i)
        new_ent = p * entropy(y[x_i >= thred]) + \
            (1 - p) * entropy(y[x_i < thred])
        if new_ent < min_entropy:
            min_entropy = new_ent
            threshold = thred
    return res - min_entropy, threshold


def is_pure(y):
    return len(set(y)) == 1


def recursive_split(x, y):
    # if all labels is the same in the sub data set , return
    if is_pure(y) or len(y) == 0:
        return y
    # number of feature
    num_feature = x.shape[1]

    # calculate information gain of each x attribute
    gains = []
    for i in range(num_feature):
        if is_discrete[i]:
            gains.append(info_gain_discrete(x[:, i], y))
        else:
            gains.append(info_gain_continue(x[:, i], y))
    gains = np.array(gains)

    # select attribute with the max information gain
    selected_attr_idx = np.argmax(gains[:, 0])
    # print('selected_attr', selected_attr_idx)
    # print('select_idx_gain', gains[selected_attr_idx, :])

    # if information gains are too small
    if np.all(gains < 1e-3) or gains[selected_attr_idx, 0] < 1e-6:
        return y

    # split the data set using selected attribute
    if is_discrete[selected_attr_idx]:
        split_sets = partition_discrete(x[:, selected_attr_idx])
    else:
        split_sets = partition_continue(
            x[:, selected_attr_idx], gains[selected_attr_idx, 1])

    res = {}
    for k, v in split_sets.items():
        # split data into sub data set
        x_sub, y_sub = x.take(v, axis=0), y.take(v, axis=0)
        res["x_%d = %s" % (selected_attr_idx, k)
            ] = recursive_split(x_sub, y_sub)
    return res


data = create_data()
x, y = data[:, 1:], data[:, 0]
# whether data in column is discrete, if is_discrete[i] == True, column i
# in x is discrete
is_discrete = [True, False, False, False, False, False, True]

print('Test for entropy:', entropy([1, 2]))
print('Test for entropy:', entropy([1, 1]))

print('Test for info_gian:', info_gain_continue(
    np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])))
print('Test for info_gian:', info_gain_continue(
    np.array([1, 2, 3, 4]), np.array([0, 1, 1, 1])))

print('Test for partition:', partition_continue(
    [0.1, 0.5, 0.6, 0.3, 0.5, 0.2, 0.1], 0.3))

print("\nDecision Tree Structure:")
pprint(recursive_split(x, y))
