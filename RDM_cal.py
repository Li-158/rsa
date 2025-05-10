import numpy as np
from neurora.stuff import limtozero
import math
from scipy.stats import pearsonr
from neurora.stuff import show_progressbar
from neurora.decoding import tbyt_decoding_kfold

def fmriRDM(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1], method="correlation", abs=False)
    '''
    fmri_data: 
        The fmri data. 
        The shape of fmri_data is [n_cons, nx, ny, nz]
    '''
    if len(np.shape(fmri_data)) != 4:
        print("\nThe shape of input for fmriRDM() function must be [n_cons, n_subs, nx, ny, nz].\n")
        return "Invalid input!"
    
    cons, nx, ny, nz = np.shape(fmri_data)

    # The size of the calculation units for searchlight
    # 每一次的 searchlight 大小
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    # 設定每一次移動多少距離
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]   
       
    # calculate the number of the calculation units in the x, y, z directions 
    # 計算需要跑幾個單位
    n_x = int((nx - kx)/ sx) +1 
    n_y = int((ny - ky)/ sy) +1 
    n_z = int((nz - kz)/ sz) +1 

    # Initialize the data for calculating the RDM
    data = np.full([n_x, n_y, n_z, cons, kx*ky*kz], np.nan)

    print("\nComputing RDMs")

    # assignment
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                for i in range(cons):

                    index = 0

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):
                                    data[x, y, z, i, index] = fmri_data[i, x*sx+k1, y*sy+k2, z*sz+k3]

                                index = index + 1
    # flatten the data for different calculating conditions
    data = np.reshape(data, [n_x, n_y, n_z, cons, kx*ky*kz])

    # initialize the RDMs
    subrdms = np.full([n_x, n_y, n_z, cons, cons], np.nan)

    total = n_x * n_y * n_z

    for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):

                    # show the progressbar
                    percent = (n_x * n_y * n_z + x * n_y * n_z + y * n_z + z + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    for i in range(cons):
                        for j in range(cons):

                            # no NaN
                            if (np.isnan(data[:, x, y, z, i]).any() == False) and \
                                    (np.isnan(data[:, x, y, z, j]).any() == False):
                                if method == 'correlation':
                                    # calculate the Pearson Coefficient
                                    r = pearsonr(data[x, y, z, i], data[x, y, z, j])[0]
                                    # calculate the dissimilarity
                                    if abs == True:
                                        subrdms[x, y, z, i, j] = limtozero(1 - np.abs(r))
                                    else:
                                        subrdms[x, y, z, i, j] = limtozero(1 - r)
                                elif method == 'euclidean':
                                    subrdms[x, y, z, i, j] = np.linalg.norm(data[x, y, z, i] -
                                                                                 data[x, y, z, j])
                                """elif method == 'mahalanobis':
                                    X = np.transpose(np.vstack((data[sub, x, y, z, i], data[sub, x, y, z, j])), (1, 0))
                                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                                    subrdms[sub, x, y, z, i, j] = np.linalg.norm(X[:, 0] - X[:, 1])"""
                    if method == 'euclidean':
                        max = np.max(subrdms[x, y, z])
                        min = np.min(subrdms[x, y, z])
                        subrdms[x, y, z] = (subrdms[x, y, z] - min) / (max - min)

    # average the RDMs
    rdms = np.average(subrdms, axis=0)
