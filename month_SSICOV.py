import os
import sys
import time

import numpy as np
import pandas as pd

from SsiCoV import SSICOV

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('''Usage:
                    python3 month_SSICOV csv_file_path save_csv_path''')
        sys.exit()
    # chunk size = 20min data
    df = pd.read_csv(sys.argv[1], chunksize=60000)
    fn_fd = sys.argv[2]
    if os.path.exists(fn_fd):
        os.remove(fn_fd)
    model = SSICOV(dt=.02, min_order=20, max_order=40)
    for index, chunk in enumerate(df):
        start_time = time.time()
        data = chunk.values
        h = index // 3
        m = index % 3 * 20
        fn, _, _ = model.fit(data)
        with open(fn_fd, 'a') as fn_file:
            fn = np.array([fn])
            print(fn.shape)
            np.savetxt(fn_file, fn, delimiter=',')
        end_time = time.time()
        print('Data From %02d:%02d processed' % (h, m))
        print('The processing time: %.2f' % (end_time - start_time))
