import numpy as np
import math
import pandas as pd
import nibabel as nib
import os
from neurora.rdm_cal import bhvRDM, eegRDM, fmriRDM
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.stuff import show_progressbar, limtozero
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from neurora.nii_save import corr_save_nii
from rsa_cal import rsa

# Initialize path, name
path = '/home/.bml/projects/03_decision-space-navigation/projects/03-08_decision-space-navigation-age-distance-est/data'
'''
folder_list = [
    'MYP001', 'MYP002', 'MYP003',
    'Y001', 'Y002', 'Y003', 'Y004', 'Y005', 'Y006', 'Y007',
    'Y008', 'Y009', 'Y010', 'Y011', 'Y012', 'Y013', 'Y014',
    'Y015', 'Y016', 'Y017', 'Y018', 'Y019', 'Y020', 'Y021']

 folder_list = [
    'MOP001', 'MOP002', 'MOP004',
    'O002', 'O004', 'O007', 'O008', 'O009',
    'O011', 'O013', 'O019', 'O020', 'O021', 'O022',
    'O023', 'O029', 'O030', 'O031', 'O033', 'O034',
    'O035', 'O037', 'O038', 'O039'
]
'''
for name in folder_list:
    # loading the fmri data name
    file_list=[]
    for run in range(1,9):  # In this paradigm, it contain 8 run and 7 trial/run.
        for trial in range(1,8):
            file_name=os.path.join(path, 'derivatives', f'sub-{name}','trial_beta_judgment',
                                    f'sub-{name}_run-00{run}_trial-0{trial}_beta.nii')
            file_list.append(file_name)
    
    print(f'Loading fMRI data:{name}')
    # loading fmri data 
    fmri_img=[]
    for file in file_list:
        img = nib.load(file)
        fdata = img.get_fdata()
        fmri_img.append(fdata)

    # reshape the data dimension to match the following function 
    fmri_data = np.stack(fmri_img, axis=0)
    
    print(f'Loading behavior data:{name}')
    # load the behavior data 
    beh = pd.read_csv(os.path.join(path, 'derivatives', 'beh', '1_total_info.csv'), encoding='big5')
    sub_data = beh[beh['ID'] == name]
    bhv_data = np.array(sub_data[['DJ']])
    bhv_data = bhv_data.reshape(56,1,1)
    
    print(f"Loading affine:{name}")
    # loading affine name and data
    affine_name = os.path.join(path, 'derivatives', f'sub-{name}','func',
                                    f'arsub-{name}_task-dsnage_run-001_bold.nii')
    affine = nib.load(affine_name).affine

    # Setting output_path and filename
    output_path = os.path.join(path, 'derivatives', f'sub-{name}',f'sub-{name}_RSA_subjective.nii')

    # Start to analyze 
    print(f'Starting analyzing: {name}')
    
    # Recall rsa function from rsa_cal.py
    rsa_result = rsa(bhv_data, fmri_data)

    # Saving rsa result
    print(f'Save RSA result:{name}')
    corr_save_nii(rsa_result, affine, output_path, size=[76, 76, 36], plotrlt=False)