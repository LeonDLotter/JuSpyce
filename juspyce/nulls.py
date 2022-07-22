import logging

import nibabel as nib
import numpy as np
import pandas as pd
from brainsmash.mapgen import Base
from joblib import Parallel, delayed
from neuromaps.images import load_gifti, load_nifti
from neuromaps.nulls.nulls import _get_distmat
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
lgr.setLevel(logging.INFO)

def get_distance_matrix(parc, parc_space, parc_hemi=["L", "R"], 
                        parc_density="10k", centroids=False, 
                        n_cores=1, verbose=True):
    
    ## generate distance matrix
    # case volumetric 
    if parc_space in ["MNI152", "mni152", "MNI", "mni"]:
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
            for i, i_parcel in enumerate(tqdm(parcels, desc="Calculating distance matrix", 
                                              disable=not verbose)):
                j = i
                for _ in range(n_parcels - j):
                    dist[i,j] = cdist(ijk_parcels[i_parcel], ijk_parcels[parcels[j]])\
                        .mean().astype('float32')
                    j += 1
            # mirror to lower triangle
            dist = dist + dist.T
            # zero diagonal
            np.fill_diagonal(dist, 0)
    
    # case surface
    elif parc_space in ["fsaverage", "fsLR", "fsa", "fslr"]:
        
        def surf_dist(i_hemi, hemi):
            dist = _get_distmat(
                hemi, 
                atlas=parc_space, 
                density=parc_density, 
                parcellation=parc[i_hemi] if len(parc_hemi)>1 else parc,
                n_proc=n_cores)
            return(dist)
        
        n_jobs = 2 if (n_cores>1) & len(parc_hemi)>1 else 1
        dist = Parallel(n_jobs=n_jobs)(
            delayed(surf_dist)(i, h) for i, h in enumerate(tqdm(
                parc_hemi, 
                desc=f"Calculating distance matrix ({n_jobs} proc)", disable=not verbose)))
        
        if isinstance(parc, tuple):
            dist = tuple(dist)
        else:
            dist = dist[0]

    ## return
    return dist


def generate_null_maps(data, parcellation, dist_mat=None, 
                       parc_space=None, parc_hemi=None, parc_density=None, 
                       n_nulls=1000, centroids=False,
                       n_cores=1, seed=None, verbose=True):
    
    ## input data
    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        lgr.critical(f"Input data not array-like! Type: {type(data)}")
    n_data = data.shape[0]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data_labs = list(data.index)
        data = np.array(data)
    else:
        data_labs = list(range(n_data))
    if len(data.shape)==1:
        data = data[np.newaxis,:]
        
    # print
    lgr.info(f"Null map generation: Assuming n = {n_data} data vector(s) for "
             f"n = {data.shape[1]} parcels.")
    
    ## load parcellation if distance matrix is None
    if dist_mat is None:        
        
        # load function
        def load_parc(parc, parc_type, parc_space):
            if parc_type=="nifti":
                parc = load_nifti(parc)
                parc_space = "MNI152" if parc_space is None else parc_space
                n_parcels = len(np.trim_zeros(np.unique(parc.get_fdata())))
            elif parc_type=="gifti":
                parc = load_gifti(parc)
                parc_space = "fsaverage" if parc_space is None else parc_space
                n_parcels = len(np.trim_zeros(np.unique(parc.darrays[0].data)))
            elif parc_type=="giftituple":
                parc = (load_gifti(parc[0]), load_gifti(parc[1]))
                parc_space = "fsaverage" if parc_space is None else parc_space
                n_parcels = (len(np.trim_zeros(np.unique(parc[0].darrays[0].data))),
                             len(np.trim_zeros(np.unique(parc[1].darrays[0].data))))
            return parc, parc_space, n_parcels
        
        # recognize parcellation type
        if isinstance(parcellation, nib.Nifti1Image):
            parc_type = "nifti"
        elif isinstance(parcellation, nib.GiftiImage):
            parc_type = "gifti"
        elif isinstance(parcellation, tuple):
            parc_type = "giftituple"
        elif isinstance(parcellation, str):
            if parcellation.endswith(".nii") | parcellation.endswith(".nii.gz"):
                parc_type = "nifti"
            elif parcellation.endswith(".gii") | parcellation.endswith(".gii.gz"):
                parc_type = "gifti"
            else:
                lgr.critical(f"'parcellation' is string ({parcellation}) "
                             "but ending was not recognized!")
        else:
            lgr.critical("'parcellation' data type not defined!")    
        
        ## load parcellation
        parc, parc_space, n_parcels = load_parc(parcellation, parc_type, parc_space)

        # check for problems
        if isinstance(parc, nib.GiftiImage):
            if (parc_hemi is None) | (len(parc_hemi)>1):
                lgr.warning("If only one gifti parcellation image is supplied, 'parc_hemi' must "
                            "be one of: ['L'], ['R']! Assuming left hemisphere!" )
                parc_hemi = ["L"]
        if isinstance(parc, tuple):
            if (parc_hemi is None) | (len(parc_hemi)==1):
                lgr.warning("If 'parc_hemi' is ['L'] or ['R'], only one gifti parcellation image "
                            "should be supplied as string or gifti! Assuming both hemispheres!")
                parc_hemi = ["L", "R"]   
        
        # print
        temp = f", parc_hemi = {parc_hemi}, parc_density = '{parc_density}'"
        lgr.info(f"Loaded parcellation (parc_space = '{parc_space}'"
                 f"{temp if parc_space in ['fsaverage', 'fsLR', 'fsa', 'fslr'] else ''}).")
    
        ## calculate distance matrix
        lgr.info("Calculating distance matrix/matrices ({d}).".format(
            d='euclidean' if parc_space in ['mni','MNI','mni152','MNI152'] else 'geodesic'))
        dist_mat = get_distance_matrix(
            parc=parc, 
            parc_space=parc_space,
            parc_hemi=parc_hemi,
            parc_density=parc_density,
            centroids=centroids,
            n_cores=n_cores,
            verbose=False)
    
    ## distance matrix provided -> parcellation not needed
    else:
        lgr.info(f"Using input distance matrix/matrices.")
        parc = None
        if len(dist_mat)==1:
            n_parcels = dist_mat[0].shape[0]
            if parc_space is None:
                lgr.warning("Distance matrix provided but 'parc_space' is None: "
                            "Assuming 'MNI152'!")
                parc_space = "MNI152"
        else:
            n_parcels = (dist_mat[0].shape[0],
                         dist_mat[1].shape[0])     
            if parc_space is None:
                lgr.warning("Distance matrix provided but 'parc_space' is None: "
                            "Assuming 'fsaverage'!")
                parc_space = "fsaverage"
      
    # check for problems          
    if np.sum(n_parcels)!=data.shape[1]:
        lgr.critical(f"Number of parcels in data (1. dimension, {data.shape[1]}) "
                        f"does not match number of parcels in parcellation ({n_parcels})!")
    
    ## generate null data
    nulls = dict()
    for i, i_lab in enumerate(tqdm(data_labs, desc=f"Generating null maps ({n_cores} proc)", 
                                   disable=not verbose)):
        nulls[i_lab] = np.zeros((n_nulls, int(np.sum(n_parcels))))
        
        # case distance matrix is array -> volumetric or one surface
        if isinstance(dist_mat, np.ndarray):
            null_data = np.full((n_nulls, n_parcels), np.nan)
            hdist = dist_mat
            hdata = np.squeeze(data[i,:])
            med = np.isinf(hdist + np.diag([np.inf] * len(hdist))).all(axis=1)
            mask = np.logical_not(np.logical_or(np.isnan(hdata), med))
            # null data
            generater = Base(
                x=hdata[mask], 
                D=hdist[np.ix_(mask, mask)], 
                seed=seed,
                n_jobs=n_cores)
            null_data[:,mask] = generater(n_nulls, 100)
            nulls[i_lab] = null_data

        # case tuple of distance matrices -> two surface hemispheres
        elif isinstance(dist_mat, tuple):
            for i_hemi, idx in enumerate([slice(0, n_parcels[0]), 
                                          slice(n_parcels[0], int(np.sum(n_parcels)))]):
                null_data = np.full((n_nulls, n_parcels[i_hemi]), np.nan)
                hdist = dist_mat[i_hemi]
                hdata = np.squeeze(data[i,idx])
                med = np.isinf(hdist + np.diag([np.inf] * len(hdist))).all(axis=1)
                mask = np.logical_not(np.logical_or(np.isnan(hdata), med))
                generater = Base(
                    x=hdata[mask], 
                    D=hdist[np.ix_(mask, mask)],
                    seed=seed,
                    n_jobs=n_cores)
                null_data[:,mask] = generater(n_nulls, 100)
                nulls[i_lab][:,idx] = null_data
        
        # case error
        else:
            lgr.critical("Distance matrix is wrong data type, should be array or tuple of arrays, "
                         f"is: {type(dist_mat)}!")

    ## return
    lgr.info("Null data generation finished.")
    return nulls, dist_mat
        
    