# refer: http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

import numpy as np
from pprint import pprint

print("Decision Tree in Python.")


def create_data():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [1, 0, 0],
                     [1, 1, 0], [2, 1, 1], [2, 0, 0]])


def entropy(y):
    # H = sum(-p * log2(p))
    res = 0.0
    unique, counts = np.unique(y, return_counts=True)
    freqs = counts.astype('float') / len(y)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def gini(y):
    res = 1.0
    unique, counts = np.unique(y, return_counts=True)
    freqs = counts.astype('float') / len(y)
    for p in freqs:
        res -= p * p
    return res


def partition(x_i):
    return {val: (x_i == val).nonzero()[0] for val in np.unique(x_i)}


def info_gain(x_i, y, type='entropy'):
    # I(y,x)=H(y)−[px=0 H(y|x=0)+px=1 H(y|x=1))]
    if type == 'entropy':
        res = entropy(y)
    elif type == 'gini':
        res = gini(y)
    values, counts = np.unique(x_i, return_counts=True)
    freqs = counts.astype('float') / len(x_i)
    if type == 'entropy':
        for val, p in zip(values, freqs):
            res -= p * entropy(y[x_i == val])
    elif type == 'gini':
        for val, p in zip(values, freqs):
            res -= p * gini(y[x_i == val])
    return res


def is_pure(y):
    return len(set(y)) == 1


def recursive_split(x, y, type='entropy'):
    # if all labels is the same in the sub data set , return
    if is_pure(y) or len(y) == 0:
        return y

    # calculate information gain of each x attribute
    gains = np.array([info_gain(x_i, y, type) for x_i in x.T])

    # select attribute with the max information gain
    selected_attr = np.argmax(gains)

    # if information gains are too small
    if np.all(gains < 1e-5):
        return y

    # split the data set using selected attribute
    split_sets = partition(x[:, selected_attr])

    res = {}
    for k, v in split_sets.items():
        # split data into sub data set
        x_sub, y_sub = x.take(v, axis=0), y.take(v, axis=0)
        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_sub, y_sub)
    return res


data = create_data()
x = data[:, :-1]
y = data[:, -1]

print('Test for entropy:', entropy([1, 2]))
print('Test for entropy:', entropy([1, 1]))

print('Tst for info_gian:', info_gain(np.array([3, 3, 4]), np.array([0, 0, 1])))
print('Tst for info_gian:', info_gain(np.array([2, 2, 2]), np.array([0, 0, 1])))

data_sub = partition(data[:, 0])
print('Test for partition:', partition([0, 1, 1, 2]))

print("\nDecision Tree Structure(entropy):")
pprint(recursive_split(x, y))

print("\nDecision Tree Structure(gini):")
pprint(recursive_split(x, y, 'gini'))
