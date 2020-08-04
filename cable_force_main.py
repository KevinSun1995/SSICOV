import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from SsiCoV import SSICOV


def engine(data):
    start_time = time.time()
    model = SSICOV(dt=.02, min_order=30, max_order=50)
    fn, x, y = model.fit(data)
    diff_list = []
    for i in range(len(fn) - 1):
        first = fn[i]
        second = fn[i + 1]
        diff_list.append(second - first)
    n = len(diff_list)
    pos = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pos[i][j] = abs(diff_list[i] - diff_list[j]) / diff_list[i]
    myClus = AgglomerativeClustering(linkage='single', affinity='euclidean',
                                     n_clusters=5).fit_predict(pos)
    Ncluster = np.max(myClus) + 1

    diff_group = []
    for r in range(Ncluster):
        count = 0
        index_list = []
        diff_c = []
        for index, value in enumerate(myClus):
            if value == r:
                count += 1
                index_list.append(index)
        if count > 3:
            for index in index_list:
                diff_c.append(diff_list[index])
            diff_group.append(diff_c)
    print('The identified frequency difference:')
    result = []
    for g in diff_group:
        r = sum(g) / len(g)
        print(r)
        result.append(r)
    print('The running time: {}'.format(time.time() - start_time))
    print(result)
    return result


if __name__ == '__main__':
    df = pd.read_csv('cable_force.csv', 'r').values
    fn_list = []
    for hour in range(24):
        for minute in [0, 20, 40]:
            s = hour * 180000 + 3000 * minute
            r = engine(df[s: s + 60000])
            print('{}:{} data processed'.format(hour, minute))
            fn_list.append(r)
    with open('cable_force_result.txt', 'w') as f:
        for f_l in fn_list:
            if len(f_l) > 1:
                f.write(str(min(f_l)) + ' ')
            else:
                f.write(str(f_l[0]) + ' ')
