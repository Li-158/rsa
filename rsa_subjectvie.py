# -*- coding: utf-8 -*-
from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import os 
import numpy as np
import pandas as pd
import nibabel as nib
from neurora.stuff import get_affine, datamask
from neurora.nps_cal import nps_fmri, nps_fmri_roi
from neurora.rsa_plot import plot_rdm
from neurora.rdm_cal import fmriRDM_roi, fmriRDM
from neurora.corr_cal_by_rdm import fmrirdms_corr
from neurora.nii_save import corr_save_nii
from neurora.rdm_cal import bhvRDM


# Initialize path, name
path = '/home/.bml/projects/03_decision-space-navigation/projects/03-08_decision-space-navigation-age-distance-est/data'
folder_list = ['MYP002', 'MYP003','Y001', 'Y002', 'Y003', 'Y004', 'Y005', 'Y006', 'Y007','Y008']
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

for name in folder_list:
    print(name)
    # Load the fmri data name
    file_list=[]
    for run in range(1,9):  # In this paradigm, it contain 8 run and 7 trial/run.
        for trial in range(1,8):
            file_name=os.path.join(path,'derivatives',f'sub-{name}','trial_beta_judgment',
                                  f'sub-{name}_run-00{run}_trial-0{trial}_beta.nii')
            file_list.append(file_name)

    # load the behavior data 
    beh_data = pd.read_csv(os.path.join(path,'derivatives','beh','1_total_info.csv'),encoding ='big5')
    sub_beh = beh_data[beh_data['ID'] == name]
    beh = np.array(sub_beh[['DJ']])
    beh = beh.reshape(56,1,1)

    # Loading fmri data 
    fmri_data=[]
    for file in file_list:
        img = nib.load(file)
        fdata = img.get_fdata()
        fmri_data.append(fdata)
    # reshape the data dimension to match the following function 
    fmri_img = np.stack(fmri_data, axis=0)
    fmri_img = np.expand_dims(fmri_img, axis=1)

    # Calculate the fmri RDM 
    fmri_rdm = fmriRDM(fmri_img,sub_opt=1, method='correlation', abs=False)
    
    # Calculate the behavior RDM
    beh_rdm = bhvRDM(beh)

    # Correlate fmri RDM with Behaivior RDM -> RSA 
    print('Calculate RSA')
    rsa_result = fmrirdms_corr(beh_rdm[0], fmri_rdm[0], method='pearson')

    # Get the affine (the matrix to convert into MNI space) 
    affine_name = os.path.join(path,'derivatives',f'sub-{name}','func',f'arsub-{name}_task-dsnage_run-001_bold.nii')
    affine = get_affine(affine_name)

    # RSA file save to .nii file 
    result_dir = os.path.join(path, 'derivatives',f'sub-{name}',f'sub-{name}_rsa_subjective_img.nii')
    rsaresult = corr_save_nii(rsa_result, filename=result_dir, affine=affine, size=[78, 78, 38], p=0.05, correct_method='Cluster-FDR', plotrlt=False)
    print(f'Finish the calculation of RSA: {name}')
