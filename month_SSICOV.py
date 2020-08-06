import os
import sys
import time
from shutil import rmtree

import numpy as np
import pandas as pd

from SsiCoV import SSICOV

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('''Usage:
                    python3 month_SSICOV csv_file_path save_csv_path''')
        sys.exit()
    # chunk size = 20min data
    df = pd.read_csv(sys.argv[1], chunksize=180000)
    result_folder = sys.argv[2]
    if os.path.exists(result_folder):
        rmtree(result_folder)
    os.mkdir(result_folder)
    model = SSICOV(dt=.02, min_order=20, max_order=40)
    for index, chunk in enumerate(df):
        start_time = time.time()
        data = chunk.values
        hour = index
        day = index // 24
        fn, zeta, _ = model.fit(data)
        fn_fd = os.path.join(result_folder, 'fn.csv')
        zeta_fd = os.path.join(result_folder, 'zeta.csv')
        with open(fn_fd, 'a') as fn_file:
            fn = np.array([fn])
            np.savetxt(fn_file, fn, delimiter=',')
        with open(zeta_fd, 'a') as zeta_file:
            zeta = np.array([zeta])
            np.savetxt(zeta_file, zeta, delimiter=',')
        end_time = time.time()
        print('Data From Day:%02d Hour:%02d processed' % (day, hour))
        print('The processing time: %.2f' % (end_time - start_time))
