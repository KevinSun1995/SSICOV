import sys

import matplotlib.pyplot as plt
import numpy as np


class Node(object):
    """
    row: int
    n_cluster: int
    forge: bool
    value: float
    """


with open('fn_csv') as f:
    dataset = []
    first_array = []
    n_order = 10
    size = 0
    for line in f.readlines():
        data = [float(x) for x in line.strip().split(',')[:n_order]]
        dataset.append(data)
        size += 1
    dataset = np.array(dataset)
    for c in range(dataset.shape[1]):
        c_data = dataset[:, c]
        plt.scatter(range(dataset.shape[0]), c_data)
    plt.show()
    plt.close()

    a_data = np.zeros((size, n_order), dtype=float)

    for c in range(n_order):
        c_data = dataset[:, c]
        sorted_data = np.sort(c_data)
        s_data = sorted_data[int(size * .25):int(size * .75)]
        d_mean = np.mean(c_data)
        d_d = np.std(c_data)
        s_mean = np.mean(s_data)
        r = 0
        while r < size:
            z = (c_data[r] - s_mean)
            # omit one order
            if z > 3 * d_d:
                if c < n_order - 1:
                    dataset[r][c + 1:] = dataset[r][c:-1]
                c_data[r] = s_mean
                r -= 1

            # recognize extra order
            elif z < -3 * d_d:
                if c < n_order - 1:
                    dataset[r][c:-1] = dataset[r][c + 1:]
                    dataset[r][-1] = np.mean(dataset[:, -1])
                r -= 1
            r += 1

    for c in range(dataset.shape[1]):
        c_data = dataset[:, c]
        plt.scatter(range(dataset.shape[0]), c_data)
    plt.show()
    plt.close()

    # c = len(dataset)
    # n_cluster = 5
    # plt_matrix = np.zeros((c, n_cluster), dtype=Node)
    #
    # flatten_data = []
    # len_list = []
    # for d in dataset:
    #     flatten_data += d
    #     len_list.append(len(d))

    # n_element = len(flatten_data)
    # pos = np.zeros((n_element, n_element))
    # for i in range(n_element):
    #     for j in range(n_element):
    #         pos[i][j] = abs(flatten_data[i] - flatten_data[j]) / flatten_data[i]
    #
    # myClus = AgglomerativeClustering(linkage='complete', affinity='euclidean',
    #                                  n_clusters=n_cluster).fit_predict(pos)
    # Ncluster = np.max(myClus) + 1
    # print(flatten_data[:5])
    # print(flatten_data[5:10])
    # print(myClus[:5])
    # print(myClus[5:10])
    sys.exit()
