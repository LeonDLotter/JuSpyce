import logging

import nibabel as nib
import numpy as np
import pandas as pd
from brainsmash.mapgen import Base
from neuromaps.images import load_gifti, load_nifti
from neuromaps.nulls import burt2020
from neuromaps.nulls.nulls import _get_distmat
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
lgr.setLevel(logging.INFO)

def get_distance_matrix(parcellation, parc_space=None, parc_hemi=["L", "R"], 
                        parc_density="10k", centroids=False, 
                        n_cores=1, verbose=True):
    
   # load parcellation
    if isinstance(parcellation, (nib.Nifti1Image, str)):
        parc = load_nifti(parcellation)
        parc_space = "MNI152"
    elif isinstance(parcellation, tuple):
        parc = (load_gifti(parcellation[0]),
                load_gifti(parcellation[1]))
        parc_space="fsaverage"
    else:
        print("'parcellation' data type not defined!")
    
    ## generate distance matrix
    # case volumetric 
    if parc_space=="MNI152":
        # get parcellation data
        parc_data = parc.get_fdata()
        parc_affine = parc.affine
        parcels = np.trim_zeros(np.unique(parc_data))
        n_parcels = len(parcels)
        mask = np.logical_not(np.logical_or(np.isclose(parc_data, 0), np.isnan(parc_data)))
        parc_data_m = parc_data * mask

        # case distances between parcel centroids
        if centroids:  
            # get centroid coordinates in world space
            xyz = np.zeros((n_parcels, 3), float)
            for i, i_parcel in enumerate(parcels):
                xyz[i,:] = np.column_stack(np.where(parc_data_m==i_parcel)).mean(axis=0)
            ijk = nib.affines.apply_affine(parc_affine, xyz)
            # get distances
            dist = np.zeros((n_parcels, n_parcels), dtype='float32')
            for i, row in enumerate(ijk):
                dist[i] = cdist(row[None], ijk).astype('float32')   
            
        # case mean distances between parcel-wise voxels 
        else:
            # get parcel-wise coordinates in world space
            ijk_parcels = dict()
            for i_parcel in parcels:
                xyz_parcel = np.column_stack(np.where(parc_data_m==i_parcel))
                ijk_parcels[i_parcel] = nib.affines.apply_affine(parc_affine, xyz_parcel)
            # get distances for upper triangle of matrix
            dist = np.zeros((n_parcels, n_parcels), dtype='float32')
            for i, i_parcel in enumerate(tqdm(parcels, desc="Calculating distance matrix", disable=not verbose)):
                j = i
                for _ in range(n_parcels - j):
                    dist[i,j] = cdist(ijk_parcels[i_parcel], ijk_parcels[parcels[j]]).mean().astype('float32')
                    j += 1
            # mirror to lower triangle
            dist = dist + dist.T - np.diag(np.diag(dist))
        
    # case surface
    elif parc_space in ["fsaverage", "fsLR"]:
        dist = list()
        for i_hemi, hemi in enumerate(tqdm(parc_hemi, desc="Calculating distance matrix", disable=not verbose)):
            dist.append(_get_distmat(hemi, 
                                     atlas=parc_space, 
                                     density=parc_density, 
                                     parcellation=parcellation[i_hemi],
                                     n_proc=n_cores))
        dist = tuple(dist)

    ## return
    return dist


def generate_null_maps(data, dist_mat=None, parcellation=None, 
                       parc_space=None, parc_hemi=["L", "R"], parc_density="10k", 
                       n_nulls=1000, centroids=False,
                       n_cores=1, seed=None, verbose=True):
    
    ## load parcellation
    if dist_mat is None:
        if isinstance(parcellation, (nib.Nifti1Image, str)):
            parc = load_nifti(parcellation)
            parc_space = "MNI152" if parc_space is None else parc_space
        elif isinstance(parcellation, tuple):
            parc = (load_gifti(parcellation[0]),
                    load_gifti(parcellation[1]))
            parc_space = "fsaverage" if parc_space is None else parc_space
        else:
            lgr.error("'parcellation' data type not defined!")
    else:
        parc = parcellation
    
    ## input data
    n_data = data.shape[0]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data_labs = list(data.index)
        data = data.values
    else:
        data_labs = list(range(n_data))
    if len(data.shape)==1:
        data = data[np.newaxis,:]
    lgr.info(f"Null map generation: Assuming n = {n_data} data vector(s) for n = {data.shape[1]} parcels.")
    
    ## get distance matrix
    if dist_mat is None:
        lgr.info(f"Calculating distance matrix/matrices (space = '{parc_space}').")
        dist_mat = get_distance_matrix(parcellation=parc, 
                                       parc_space=parc_space,
                                       parc_hemi=parc_hemi,
                                       parc_density=parc_density,
                                       centroids=centroids,
                                       n_cores=n_cores,
                                       verbose=False)
    else:
        lgr.info(f"Using input distance matrix/matrices.")
    
    ## generate null data
    nulls = dict()
    for i, i_lab in enumerate(tqdm(data_labs, desc="Generating null data", disable=not verbose)):
        # case volumetric data
        if parc_space in ["MNI152", "MNI"]:
            lgr.debug(f"Generating volumetric null data {i}/{n_data} (n = {n_nulls})...")

            # null data
            generater = Base(x=data[i,:], 
                             D=dist_mat, 
                             seed=seed,
                             n_jobs=n_cores)
            nulls[i_lab] = generater(n_nulls, 100)

        # case surface data
        else:
            lgr.debug(f"Generating surface null data {i}/{n_data} (n = {n_nulls})...")
            # null data (bug in neuromaps: if distmat provided, parcellation, atlas & density should be ignored -> not the case)
            nulls[i_lab] = burt2020(data=data[i,:],
                                    parcellation=parc,
                                    atlas=parc_space,
                                    density=parc_density,
                                    n_perm=n_nulls,
                                    seed=seed,
                                    n_proc=n_cores,
                                    distmat=dist_mat).T
    
    ## return
    lgr.info("Null data generation finished.")
    return nulls, dist_mat
        
    