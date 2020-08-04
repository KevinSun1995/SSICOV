import os
import time

import numpy as np
import pandas as pd

from SsiCoV import SSICOV

if __name__ == '__main__':
    df = pd.read_csv('Qian3_N_V.csv')
    data = df[['11012', '11014', '11016', '11017', '11020']].values
    fn_fd = 'fn_csv1'
    if os.path.exists(fn_fd):
        os.remove(fn_fd)
    model = SSICOV(dt=.02, min_order=20, max_order=40)
    for h in range(24):
        for m in [0, 20, 40]:
            start_time = time.time()
            start = h * 180000 + m * 3000
            end = start + 20 * 3000
            train_data = data[start: end]
            fn, _, _ = model.fit(train_data)
            with open(fn_fd, 'a') as fn_file:
                fn = np.array([fn])
                print(fn.shape)
                np.savetxt(fn_file, fn, delimiter=',')
            end_time = time.time()
            print('Data From %02d:%02d processed' % (h, m))
            print('The processing time: %.2f' % (end_time - start_time))
