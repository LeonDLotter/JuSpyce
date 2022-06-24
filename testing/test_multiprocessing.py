
#%%

import numpy as np
from multiprocessing import Pool
from joblib import Parallel, delayed
import time

def fill_array(start_val):
    return range(start_val,start_val+10)


#%%
for i in range(1, 9):
    
    tic = time.time()
    
    if __name__ == "__main__":
        pool = Pool(processes = i)    

        pool_result = pool.map(fill_array, range(0,100000))

        pool.close()
        pool.join()

        array_2D = np.zeros((100000,10))
        for line,result in enumerate(pool_result):
            array_2D[line,:] = result
        toc = time.time()

        #print(array_2D)
        print(i, toc - tic)
        
        
# %%
for i in range(1, 9):
    
    tic = time.time()
    
    pool_result = Parallel(n_jobs=i)(delayed(fill_array)(i) for i in range(0,100000))
    
    array_2D = np.zeros((100000,10))
    for line,result in enumerate(pool_result):
        array_2D[line,:] = result
    toc = time.time()

    #print(array_2D)
    print(i, toc - tic)
        

#%%
