import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from neuromaps import images, parcellate, resampling
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
                   resampling_target="data",
                   dtype=None,
                   n_proc=1,
                   verbose=True):
    
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
                    lgr.warning("Input is single GIFTI image but 'hemi' is not given. Assuming left!")
                    parc_hemi = "left"
        elif isinstance(parcellation, nib.GiftiImage):      
            parcellation = images.load_gifti(parcellation) 
        elif isinstance(parcellation, nib.Nifti1Image):      
            parcellation = images.load_nifti(parcellation) 
        elif isinstance(parcellation, tuple):
            parcellation = (images.load_gifti(parcellation[0]),
                            images.load_gifti(parcellation[1])) 
        else:
            lgr.critical(f"Parcellation data type not recognized! ({type(parcellation)})")
          
        # catch problems
        if (data_space in ["MNI152", "MNI", "mni", "mni152"]) & \
            (parc_space not in ["MNI152", "MNI", "mni", "mni152"]) & \
            (resampling_target=="data"):
                lgr.warning("Data is in MNI space but parcellation is in surface space and "
                            "'resampling_target' is 'data'! Cannot resample surface to MNI: "
                            "Setting 'resampling_target' to 'parcellation'.")
                resampling_target = "parcellation"
            
        
        # neuromaps parcellater: can deal with str, path, nifti, gifti, tuple
        parcellater = parcellate.Parcellater(
            parcellation=parcellation, 
            space=parc_space,
            resampling_target=resampling_target,
            hemi=parc_hemi
            ).fit()
        
        # data extraction function
        def extract_data(file):
            file_parc = parcellater.transform(
                data=file, 
                space=data_space)
            return file_parc
        # extract data (in parallel)
        data_parc = Parallel(n_jobs=n_proc)(delayed(extract_data)(f) for f in tqdm(
            data, desc=f"Parcellating imaging data ({n_proc} proc)", disable=not verbose))
        # collect data
        if isinstance(parcellation, tuple):
            data_parc = np.array([d for d in data_parc], dtype=dtype)
        else:  
            data_parc = np.array([d[0,:] for d in data_parc], dtype=dtype)

        # output dataframe
        if data_labels is None:
            try:
                if isinstance(data[0], tuple):
                    data_labels = [os.path.basename(f[0]).replace(".gii","").replace(".gz","") \
                        for f in data]
                else:
                    data_labels = [os.path.basename(f).replace(".nii","").replace(".gz","") \
                        for f in data]
            except:
                data_labels = list(range(len(data)))
        df_parc = pd.DataFrame(
            data=data_parc, 
            index=data_labels,
            columns=parc_labels)
    
    ## case array
    elif isinstance(data, np.ndarray):
        lgr.info("Input type: ndarray, assuming parcellated data with shape "
                 "(n_files/subjects/etc, n_parcels).")
        if len(data.shape)==1:
            data = data[np.newaxis,:]
        df_parc = pd.DataFrame(
            data=data,
            index=data_labels,
            columns=parc_labels)
            
    ## case dataframe
    elif isinstance(data, pd.DataFrame):
        lgr.info("Input type: DataFrame, assuming parcellated data with shape "
                 "(n_files/subjects/etc, n_parcels).")
        df_parc = pd.DataFrame(
            data=data.values,
            index=data_labels if data_labels is not None else data.index,
            columns=parc_labels if parc_labels is not None else data.columns)
    
    ## case series
    elif isinstance(data, pd.Series):
        lgr.info("Input type: Series, assuming parcellated data with shape (1, n_parcels).")
        df_parc = pd.DataFrame(
            data=data.values,
            index=data_labels if data_labels is not None else data.index,
            columns=parc_labels if parc_labels is not None else [data.name])
        df_parc = df_parc.T
    
    ## case not defined
    else:
        lgr.critical(f"Can't import from data with type {type(data)}!")
        
    ## check for nan's
    if df_parc.isnull().any(None):
        lgr.warning("Data contains nan's! Will be excluded on case-wise basis.")
 
    ## return data array
    return df_parc.astype(dtype)
