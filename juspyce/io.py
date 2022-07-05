import logging

import numpy as np
import pandas as pd
import os
import nibabel as nib
from neuromaps import parcellate, resampling, images
from tqdm.auto import tqdm

lgr = logging.getLogger(__name__)
lgr.setLevel(logging.INFO)

def get_input_data(data, 
                   data_labels=None,
                   data_space=None, 
                   parcellation=None, 
                   parc_labels=None,
                   parc_space=None,
                   parc_hemi=None,
                   dtype=None):
    
    ## case list
    if isinstance(data, list):
        lgr.info("Input type: list, assuming imaging data.")

        # load parcellation
        if isinstance(parcellation, str):
            if parcellation.endswith(".nii") or parcellation.endswith(".nii.gz"):
                parcellation = images.load_nifti(parcellation)
            elif parcellation.endswith(".gii") or parcellation.endswith(".gii.gz"):
                parcellation = images.load_gifti(parcellation)
                if parc_hemi is None:
                    lgr.error("Input is single GIFTI image but 'hemi' is not given. Assuming left!")
                    parc_hemi = "left"
        if isinstance(parcellation, tuple):
            parcellation = (images.load_gifti(parcellation[0]),
                            images.load_gifti(parcellation[1]))        
                
        # neuromaps parcellater: can deal with str, path, nifti, gifti, tuple
        parcellater = parcellate.Parcellater(parcellation=parcellation, 
                                             space=parc_space,
                                             resampling_target="data",
                                             hemi=parc_hemi
                                             ).fit()
        
        # extract data
        data_parc = list()
        for file in tqdm(data, desc="Parcellating imaging data"):
            file_parc = parcellater.transform(file, data_space)[0,:]
            data_parc.append(file_parc)
        data_parc = np.array(data_parc, dtype=dtype)
        # output dataframe
        if data_labels is None:
            try:
                data_labels = [os.path.basename(f) for f in data]
            except:
                data_labels = list(range(len(data)))
        df_parc = pd.DataFrame(data=data_parc, 
                               index=data_labels,
                               columns=parc_labels)
    
    ## case array
    elif isinstance(data, np.ndarray):
        lgr.info("Input type: ndarray, assuming parcellated data with shape (n_files/subjects/etc, n_parcels).")
        if len(data.shape)==1:
            data = data[np.newaxis,:]
        df_parc = pd.DataFrame(data=data,
                               index=data_labels,
                               columns=parc_labels)
            
    ## case dataframe
    elif isinstance(data, pd.DataFrame):
        lgr.info("Input type: DataFrame, assuming parcellated data with shape (n_files/subjects/etc, n_parcels).")
        df_parc = pd.DataFrame(data=data.values,
                               index=data_labels if data_labels is not None else data.index,
                               columns=parc_labels if parc_labels is not None else data.columns)
    
    ## case series
    elif isinstance(data, pd.Series):
        lgr.info("Input type: Series, assuming parcellated data with shape (1, n_parcels).")
        df_parc = pd.DataFrame(data=data.values,
                               index=data_labels if data_labels is not None else data.index,
                               columns=parc_labels if parc_labels is not None else [data.name])
        df_parc = df_parc.T
    
    ## case not defined
    else:
        lgr.error(f"Can't import from data with type {type(data)}!")
        
    ## check for nan's
    if df_parc.isnull().any(None):
        lgr.warning("Data contains nan's! Will be excluded on case-wise basis.")
 
    ## return data array
    return df_parc.astype(dtype)
