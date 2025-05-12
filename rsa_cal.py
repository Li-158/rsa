import numpy as np
import math
import pandas as pd
import nibabel as nib
from neurora.rdm_cal import bhvRDM, eegRDM, fmriRDM
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.stuff import show_progressbar
from neurora.stuff import limtozero
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from neurora.stuff import permutation_corr

def rsa(bhv_data, fmri_data, ksize=[3,3,3], strides=[1, 1, 1], use_abs=False):
    '''
    Calcualte Representation smiliarity analysis based on fMRI data (searchlight)
    
    Parameters
    ----------
    bhv_data : array.
        The behavior data. The code from neurora code need to be imported before use it.
        The shape of bhv_data must be [n_cons, n_sub, n_trial]
    fmri_data : array.
        The fmri data
        the shape of fmri_data must be [n_cons, nx, ny, nz].
        n_cons, nx, ny, nz represent the number of conditions, the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
        kx, ky, kz should be odd. if your data is from 7-T MRI, maybe modify to [1, 1, 1]
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    corrs : array
        The similarities between fMRI searchlight RDMs and a beh RDM
        The shape of RDMs is [n_x, n_y, n_z, 2]. 
        n_x, n_y, n_z represent the number of calculation units for searchlight
        along the x, y, z axis and 2 represents a r-value and a p-value.
    '''
    # Loading the behavior RDM
    bhv_rdm = bhvRDM(bhv_data)
    bhv_rdm = bhv_rdm[0]

    # get the number of conditions, subjects and the size of the fMRI-img
    cons, nx, ny, nz = np.shape(fmri_data)

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units in the x, y, z directions
    # The calculation units indicate the number of box (3*3*3)
    n_x = int((nx - kx) / sx)+1
    n_y = int((ny - ky) / sy)+1
    n_z = int((nz - kz) / sz)+1

    # initialize the data for calculating the RDM
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
                                # shape: [n_x, n_y, n_z, cons, kx*ky*kz], kx*ky*kz is the size of searchlight(27)
                                data[x, y, z, i, index] = fmri_data[i, x*sx+k1, y*sy+k2, z*sz+k3]

                                index = index + 1

    # initialize the RDMs
    subrdms = np.full([n_x, n_y, n_z, cons, cons], np.nan)

    # initialize the RSA 
    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    # to show the percent in 'show_progressbar' function
    total = n_x * n_y * n_z 
    
    print(f'Finishing assignment:{name}')

    # Generate the fMRI RDM
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                # show the progressbar
                percent = (x * n_y * n_z + y * n_z + z + 1) / total * 100
                show_progressbar("Calculating RSA", percent)

                # Calculate the similarity between searchlight. -> fMRI RDM
                for i in range(cons):
                    for j in range(cons):
                    
                        # the restrict for pearsonr is not allow nan in the followly calcualting data. 
                        if (np.isnan(data[x, y, z, i]).any() == False) and \
                            (np.isnan(data[x, y, z, j]).any() == False):
                            r = pearsonr(data[x, y, z, i], data[x, y, z, j])[0]
                            subrdms[x,y,z,i,j] = limtozero(1 - np.abs(r) if use_abs else 1-r)
               
                # Correlate fmri rdm with bhv rdm (searchlight)
                # 與行為 RDM 比較（若當前 voxel 的 fmri_rdm 完整）
                fmri_rdm = subrdms[x, y, z]
                if not np.isnan(fmri_rdm).any():

                    # corrs[x, y, z] = rdm_correlation_pearson(bhv_rdm[0], fmri_rdm)
                    # calculate the number of value above the diagonal in RDM
                    # 因為matric是對稱的，因此計算的只需要一半就好
                    n = int(cons*(cons-1/2))

                    # initialize two vectors to store the values above the diagnal of two RDMs
                    v1 = np.zeros([n])
                    v2 = np.zeros([n])

                    # assignment the variable to calculate the rsa
                    nn = 0
                    for i in range(cons - 1):
                        for j in range(cons -1 -i):
                            v1[nn] = bhv_rdm[i, i +j +1]
                            v2[nn] = fmri_rdm[i, i +j +1]
                            nn += 1
                    
                    # Calculate the Pearsonr Correlation
                    corrs[x, y, z] = np.array(pearsonr(v1, v2))

    print("\nRSA computing finished!")
    return corrs
