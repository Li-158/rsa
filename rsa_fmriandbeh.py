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



# Initialize path, name
path = '/home/.bml/projects/03_decision-space-navigation/projects/03-08_decision-space-navigation-age-distance-est/data'
name = ['MYP002']
'''
folder_list = [
    'MYP001', 'MYP002', 'MYP003',
    'Y001', 'Y002', 'Y003', 'Y004', 'Y005', 'Y006', 'Y007',
    'Y008', 'Y009', 'Y010', 'Y011', 'Y012', 'Y013', 'Y014',
    'Y015', 'Y016', 'Y017', 'Y018', 'Y019', 'Y020', 'Y021']

 folder_list2 = [
    'MOP001', 'MOP002', 'MOP004',
    'O002', 'O004', 'O007', 'O008', 'O009',
    'O011', 'O013', 'O019', 'O020', 'O021', 'O022',
    'O023', 'O029', 'O030', 'O031', 'O033', 'O034',
    'O035', 'O037', 'O038', 'O039'
]
'''

    # Load the fmri data name
file_list=[]
for run in range(1,9):  # In this paradigm, it contain 8 run and 7 trial/run.
    for trial in range(1,8):
        file_name=os.path.join(path,'derivatives',f'sub-{name}','trial_beta_judgment',
                                f'sub-{name}_run-00{run}_trial-0{trial}_beta.nii')
        file_list.append(file_name)

# Loading fmri data 
fmri_data=[]
for file in file_list:
    img = nib.load(file)
    fdata = img.get_fdata()
    fmri_data.append(fdata)

# reshape the data dimension to match the following function 
# fmri_img dims: [n_con, n_sub, nx, ny, nz]
fmri_img = np.stack(fmri_data, axis=0)
fmri_img = np.expand_dims(fmri_img, axis=1)

# load the behavior data 
# beh dims: [n_con, n_sub, n_trial]
beh_data = pd.read_csv(os.path.join(path,'derivatives','beh','1_total_info.csv'),encoding ='big5')
sub_beh = beh_data[beh_data['ID'] == name]
beh = np.array(sub_beh[['DJ']])
beh = beh.reshape(56,1,1)

import numpy as np
import math
import pandas as pd
import nibabel as nib
import os 
from neurora.rdm_cal import bhvRDM, eegRDM, fmriRDM
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.stuff import show_progressbar
from neurora.stuff import limtozero
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from neurora.stuff import permutation_corr
from scipy.stats import pearsonr



# Initialize path, name
path = '/home/.bml/projects/03_decision-space-navigation/projects/03-08_decision-space-navigation-age-distance-est/data'
name = 'MYP002'
'''
folder_list = [
    'MYP001', 'MYP002', 'MYP003',
    'Y001', 'Y002', 'Y003', 'Y004', 'Y005', 'Y006', 'Y007',
    'Y008', 'Y009', 'Y010', 'Y011', 'Y012', 'Y013', 'Y014',
    'Y015', 'Y016', 'Y017', 'Y018', 'Y019', 'Y020', 'Y021']

 folder_list2 = [
    'MOP001', 'MOP002', 'MOP004',
    'O002', 'O004', 'O007', 'O008', 'O009',
    'O011', 'O013', 'O019', 'O020', 'O021', 'O022',
    'O023', 'O029', 'O030', 'O031', 'O033', 'O034',
    'O035', 'O037', 'O038', 'O039'
]
'''

# Load the fmri data name
file_list=[]
for run in range(1,9):  # In this paradigm, it contain 8 run and 7 trial/run.
    for trial in range(1,8):
        file_name=os.path.join(path,'derivatives',f'sub-{name}','trial_beta_judgment',
                                f'sub-{name}_run-00{run}_trial-0{trial}_beta.nii')
        file_list.append(file_name)

# Loading fmri data 
fmri_img=[]
for file in file_list:
    img = nib.load(file)
    fdata = img.get_fdata()
    fmri_img.append(fdata)

# reshape the data dimension to match the following function 
# fmri_img dims: [n_con, n_sub, nx, ny, nz]
fmri_data = np.stack(fmri_img, axis=0)

# load the behavior data 
# beh dims: [n_con, n_sub, n_trial]
record = pd.read_csv(os.path.join(path,'derivatives','beh','1_total_info.csv'),encoding ='big5')
sub_beh = record[record['ID'] == name]
beh_data = np.array(sub_beh[['DJ']])
beh_data = beh_data.reshape(56,1,1)

def rsa(bhv_data, fmri_data, ksize=[3,3,3], strides=[1, 1, 1], use_abs=False):
    '''
    Calcualte Representation smiliarity analysis based on fMRI data (searchlight)
    
    I don't use the permutation test. 

    Parameters
    ----------
    bhv_data : array.
        The behavior data
        The shape of bhv data must be [n_cons, ]
    fmri_data : array.
        The fmri data
        the shape of fmri_data must be [n_cons, nx, ny, nz].
        n_cons, nx, ny, nz represent the number of conditions, the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
        kx, ky, kz should be odd.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    corrs : array
        The similarities between fMRI searchlight RDMs and a beh RDM
        The shape of RDMs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for searchlight
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
                                # [n_x, n_y, n_z, cons, kx*ky*kz]
                                data[x, y, z, i, index] = fmri_data[i, x*sx+k1, y*sy+k2, z*sz+k3]

                                index = index + 1

    # initialize the RDMs
    subrdms = np.full([n_x, n_y, n_z, cons, cons], np.nan)

    # Initialize the RSA 
    corrs = np.full([n_x, n_y, n_z, 2], np.nan)
    # To show the percent
    total = n_x * n_y * n_z 
    
    print(f'Finishing assignment:{name}')

    # Generate the fMRI RDM
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                # show the progressbar
                percent = (x * n_y * n_z + y * n_z + z + 1) / total * 100
                show_progressbar("Calculating RSA", percent)

                for i in range(cons):
                    for j in range(cons):
                    
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
                    n = int(cons*(cons-1/2))

                    # initialize two vectors to store the values above the diagnal of two RDMs
                    v1 = np.zeros([n])
                    v2 = np.zeros([n])

                    # assignment
                    nn = 0
                    for i in range(cons - 1):
                        for j in range(cons -1 -i):
                            v1[nn] = beh_rdm[i, i +j +1]
                            v2[nn] = fmri_rdm[i, i +j +1]
                            nn += 1
                    
                    # Calculate the Pearsonr Correlation
                    corr[x, y, z] = np.array(pearsonr(v1, v2))

    print("\nRSA computing finished!")
    return corrs


def save_rsa_nifti(corrs, reference_img_path, output_path, map_type="r"):
    """
    儲存 RSA 結果為 .nii 檔案

    Parameters
    ----------
    corrs : np.ndarray
        RSA 結果陣列 [n_x, n_y, n_z, 2]
    reference_img_path : str
        一張具有原始 shape 與 affine 的 fMRI NIfTI 檔案（例如原始 tmap）
    output_path : str
        輸出檔案名稱，例如 "RSA_result.nii.gz"
    map_type : str
        "r" 表儲存 r 值 (corrs[..., 0])；"p" 表儲存 p 值 (corrs[..., 1])
    """

    ref_img = nib.load(reference_img_path)
    affine = ref_img.affine
    full_shape = ref_img.shape

    # 建立一個空白影像，並把 RSA 值填進去
    rsa_map = np.zeros(full_shape)
    if map_type == "r":
        rsa_map[:corrs.shape[0], :corrs.shape[1], :corrs.shape[2]] = corrs[:,:,:,0]
    elif map_type == "p":
        rsa_map[:corrs.shape[0], :corrs.shape[1], :corrs.shape[2]] = corrs[:,:,:,1]
    else:
        raise ValueError("map_type 必須為 'r' 或 'p'")

    rsa_nifti = nib.Nifti1Image(rsa_map, affine)
    nib.save(rsa_nifti, output_path)
    print(f"儲存完成：{output_path}")

    result = rsa(beh, fmri_img)

