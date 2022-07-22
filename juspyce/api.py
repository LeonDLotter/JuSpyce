
import copy
import gzip
import logging
import os
import pickle
import sys
from telnetlib import XASCII

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pingouin import compute_effsize
from tqdm.auto import tqdm

from .datasets import get_template
from .io import get_input_data
from .nulls import generate_null_maps
from .stats import (beta, corr, dominance, mc_correction, null_to_p,
                    partialcorr3, r2, reduce_dimensions, residuals, zscore_df)
from .utils import fill_nan

logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
def set_log(verbose):
    if verbose==True:
        lgr.setLevel(logging.INFO)
        return True
    elif verbose==False:
        lgr.setLevel(logging.CRITICAL)
        return False
    else:
        lgr.setLevel(verbose)
        return True
        
            
class JuSpyce:
    """Class JuSpyce
    """   

    def __init__(self, x, y, 
                 z=None, 
                 x_labels=None, y_labels=None, z_labels=None,
                 data_space="MNI152",    
                 standardize="xy", 
                 drop_nan=False,    
                 parcellation=None, 
                 parcellation_labels=None, 
                 parcellation_space="MNI152", 
                 parcellation_hemi=["L", "R"], 
                 parcellation_density="10k",
                 resampling_target="data",
                 n_proc=1, 
                 dtype=np.float32):
        
        self.x = x
        self.y = y
        self.z = z
        self.x_lab = x_labels
        self.y_lab = y_labels
        self.z_lab = z_labels
        if isinstance(data_space, str):
            data_space = [data_space] * 3
        elif isinstance(data_space, (list, tuple)) & len(data_space)==1:
            data_space = data_space * 3
        elif isinstance(data_space, (list, tuple)) & len(data_space)==3:
            pass
        else:
            lgr.critical("'data_space' must be a string, a list with len==1 or a list with "
                         f"len==3! Is {type(data_space)} with len({len(data_space)}).")
        self.data_space = data_space
        self.parc = parcellation
        self.parc_lab = parcellation_labels
        self.parc_space = parcellation_space
        self.parc_hemi = parcellation_hemi
        self.parc_density = parcellation_density
        self.resampl_targ = resampling_target
        self.n_proc = n_proc
        self._drop_nan = drop_nan
        self._dtype = dtype
        if standardize==True:
            self.zscore = "xyz"
        elif standardize==False:
            self.zscore = ""
        else:
            self.zscore = standardize
        self._transform_count = 0
        # empty data storage dicts
        self.transforms = dict()
        self.comparisons = dict()
        self.predictions = dict()
        self.nulls = dict()
        self.p_comparisons = dict()
        self.p_predictions = dict()
    
    # ==============================================================================================
    
    def fit(self, verbose=True):
        verbose = set_log(verbose)
        
        ## extract input data
        # predictors -> usually e.g. PET atlases
        lgr.info("Checking input data for x (should be, e.g., PET data):")
        self.X = get_input_data(
            self.x, 
            data_labels=self.x_lab,
            data_space=self.data_space[0], 
            parcellation=self.parc, 
            parc_labels=self.parc_lab,
            parc_hemi=self.parc_hemi,
            parc_space=self.parc_space,
            resampling_target=self.resampl_targ,
            n_proc=self.n_proc,
            verbose=verbose,
            dtype=self._dtype)
        lgr.info(f"Got 'x' data for {self.X.shape[0]} x {self.X.shape[1]} parcels.")
        # targets -> usually e.g. subject data or group-level outcome data
        lgr.info("Checking input data for y (should be, e.g., subject data):")
        self.Y = get_input_data(
            self.y, 
            data_labels=self.y_lab,
            data_space=self.data_space[1], 
            parcellation=self.parc, 
            parc_labels=self.parc_lab,
            parc_hemi=self.parc_hemi,
            parc_space=self.parc_space,
            resampling_target=self.resampl_targ,
            n_proc=self.n_proc,
            verbose=verbose,
            dtype=self._dtype)
        lgr.info(f"Got 'y' data for {self.Y.shape[0]} x {self.Y.shape[1]} parcels.")
        # data to control correlations for
        if self.z is not None:
            lgr.info("Checking input data for z (should be, e.g., grey matter data):")
            if isinstance(self.z, str):
                if self.z in ["GM", "GMV", "gm", "gmv"]:
                    lgr.info("Using nilearn grey matter template as 'z' to control for GMV.")
                    self.z = [get_template("gm", resolution=1)]
                    self.z_lab = ["gm"]
            self.Z = get_input_data(
                self.z, 
                data_labels=self.z_lab,
                data_space=self.data_space[2], 
                parcellation=self.parc, 
                parc_labels=self.parc_lab,
                parc_hemi=self.parc_hemi,
                parc_space=self.parc_space,
                resampling_target=self.resampl_targ,
                n_proc=self.n_proc,
                verbose=verbose,
                dtype=self._dtype)
            lgr.info(f"Got 'z' data for {self.Z.shape[0]} x {self.Z.shape[1]} parcels.")
        else:
            self.Z = None
        
        ## check parcel number
        if self.X.shape[1]!=self.Y.shape[1]:
            lgr.critical("Got differing numbers of parcels in 'x' & 'y' data!")
        if self.Z is not None:
            if self.X.shape[1]!=self.Z.shape[1]:
                lgr.critical("Got differing numbers of parcels in 'x'/'y' & 'z' data!")
                  
        ## deal with nan's
        self._nan_bool = pd.concat([self.X, self.Y, self.Z], axis=0).isnull().any(axis=0)
        # case remove nan parcels completely
        if self._drop_nan==True:
            lgr.warning(f"Dropping {np.sum(self._nan_bool)} parcels with nan's. "
                        "This will lead to problems with null map generation!")
            self.X = self.X.loc[:, ~self._nan_bool]
            self.Y = self.Y.loc[:, ~self._nan_bool]
            if self.Z is not None:
                self.Z = self.Z.loc[:, ~self._nan_bool]
            self._nan_bool = pd.concat([self.X, self.Y, self.Z], axis=0).isnull().any(axis=0)
        # get column (parcel) indices and labels with nan's
        self._nan_cols = list(np.where(self._nan_bool==True)[0])
        self._nan_labs = list(self._nan_bool[self._nan_bool].index)
            
        ## parcel number
        self.n_parcels = self.X.shape[1]
        
        ## update data labels
        self.x_lab = self.X.index
        self.y_lab = self.Y.index
        
        ## z-transform
        if "x" in self.zscore:
            lgr.info("Z-standardizing 'X' data.")
            self.X = zscore_df(self.X, along="rows")
        if "y" in self.zscore:
            lgr.info("Z-standardizing 'Y' data.")
            self.Y = zscore_df(self.Y, along="rows")
        if ("z" in self.zscore) & (self.Z is not None):
            lgr.info("Z-standardizing 'Z' data.")
            self.Z = zscore_df(self.Z, along="rows")
                        
        ## return complete object
        return self

    # ==============================================================================================
    
    def transform(self, transform,
                  dataset="X", replace=False,
                  n_components=None, min_ev=None, fa_method="minres", fa_rotation="promax",
                  seed=None, store=True, verbose=True):
        verbose = set_log(verbose)
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.critical("Input data ('X', 'Y') not found. Did you run JuSpyce.fit()?!")
        
        ## get data
        data_orig = self.Y if dataset=="Y" else self.X
        data_orig_nan = data_orig.loc[:, ~self._nan_bool]
        
        ## case mean
        if transform=="mean":
            lgr.info(f"Calculating parcelwise mean of {dataset}.")
            data = np.nanmean(data_orig.values, axis=0)
            data_df = pd.DataFrame(
                data=data[np.newaxis,:], 
                index=[transform], 
                columns=data_orig.columns, 
                dtype=self._dtype)   
        
        ## case partial
        elif transform=="partial":
            lgr.info(f"Regressing 'Z' from '{dataset}': new {dataset} = residuals.")
            
            # get residuals
            data = np.zeros(data_orig_nan.shape, dtype=self._dtype)
            for xy in tqdm(range(data.shape[0]), disable=not verbose):
                data[xy,:] = residuals(
                    x=self.Z.values[:,~self._nan_bool].T,
                    y=data_orig_nan.iloc[xy,:].values.T)
            # save
            data_df = fill_nan(
                pd.DataFrame(
                    data=data, 
                    index=data_orig_nan.index,
                    columns=data_orig_nan.columns,
                    dtype=self._dtype),
                self._nan_cols, self._nan_labs, "col")
        
        ## case pca / case ICA / case fa
        elif transform in ["pca", "ica", "fa"]:
            lgr.info(f"Calculating {transform} on '{dataset}' data.")
            data, ev, loadings = reduce_dimensions(
                data=data_orig_nan.T, 
                method=transform, 
                n_components=n_components, 
                min_ev=min_ev,
                fa_method=fa_method, 
                fa_rotation=fa_rotation,
                seed=seed)
            # save
            data_df = fill_nan(
                pd.DataFrame(
                    data=data.T, 
                    index=[f"c{i}" for i in range(data.shape[1])], 
                    columns=data_orig_nan.columns, 
                    dtype=self._dtype),
                self._nan_cols, self._nan_labs, "col")
            loadings = pd.DataFrame(
                data=loadings, 
                columns=data_df.index, 
                index=data_orig.index, 
                dtype=self._dtype)
            ev = pd.Series(
                data=ev,
                index=[f"c{i}" for i in range(data.shape[1])],
                dtype=self._dtype)
        
        ## case not defined
        else:
            lgr.critical(f"transform '{transform}' not defined!")
            data_df = data_orig
            data_orig = None
        
        ## save and return     
        if store:
            self.transforms[dataset+"-"+transform] = data_df
            if transform in ["pca", "ica", "fa"]:
                self.dim_red = dict(ev=ev, loadings=loadings)
            if replace:
                lgr.info(f"Replacing '{dataset}' data with transformed data.")
                if dataset=="Y":
                    self.Y = data_df
                    self.y_lab = data_df.index
                else:
                    self.X = data_df
                    self.x_lab = data_df.index
        if transform in ["pca", "ica", "fa"]:
            return data_df, ev, loadings
        else:
            return data_df
        
    # ==============================================================================================
    
    def compare(self, comparison, groups,
                store=True, replace=False, verbose=True):
        verbose = set_log(verbose)
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.critical("Input data ('X', 'Y') not found. Did you run JuSpyce.fit()?!")
        
        # group variable
        if groups is None:
            lgr.critical("For transform '{transform}' you must provide a grouping variable!")
        groups = np.array(groups)
        idc = np.sort(np.unique(groups))
        if len(idc) > 2:
            lgr.critical("Function not defined for > 2 grouping categories!", idc)
        if len(groups) != self.Y.shape[0]:
            lgr.critical(f"Length of 'groups' ({len(groups)}) does not match length of "
                            f"Y data ({self.Y.shape[0]})!")
        # group dfs
        data_A = self.Y[groups==idc[0]]
        data_B = self.Y[groups==idc[1]]   
                
        # compare            
        ## case diff(A,mean(B))
        if comparison=="diff(A,mean(B))":
            lgr.info("Subtracting parcelwise mean of B from A: new Y = Y[A] - mean(Y[B]).")
            data = data_A - np.nanmean(data_B, axis=0)
            data_df = pd.DataFrame(
                data=data, 
                index=data_A.index, 
                columns=data_A.columns, 
                dtype=self._dtype)  

        ## case diff(B,mean(A))
        elif comparison=="diff(B,mean(A))":
            lgr.info("Subtracting parcelwise mean of A from B: new Y = Y[B] - mean(Y[A]).")
            data = data_B - np.nanmean(data_A, axis=0)
            data_df = pd.DataFrame(
                data=data, 
                index=data_B.index, 
                columns=data_B.columns, 
                dtype=self._dtype)  
            
        ## case diff(mean(A),mean(B))
        elif comparison=="diff(mean(A),mean(B))":
            lgr.info("Subtracting parcelwise mean of B from mean of A: "
                     "new Y = mean(Y[A]) - mean(Y[B]).")
            data = np.nanmean(data_A, axis=0) - np.nanmean(data_B, axis=0)
            data_df = pd.DataFrame(
                data=data[np.newaxis,:], 
                index=[comparison], 
                columns=data_A.columns, 
                dtype=self._dtype)
        
        # case cohen(A,B) / case hedge(A,B) / case pairedcohen(A,B)
        elif comparison in ["cohen(A,B)", "hedge(A,B)", "pairedcohen(A,B)"]:
            es = "cohen" if "cohen" in comparison else "hedges"
            pair = True if "paired" in comparison else False
            lgr.info(f"Calculating parcelwise effect size between A and B ({es}, paired: {pair}).")
            data = np.zeros(self.n_parcels, dtype=self._dtype)
            for p in tqdm(range(self.n_parcels), disable=not verbose):
                data[p] = compute_effsize(
                    x=data_A.iloc[:,p].values,
                    y=data_B.iloc[:,p].values,
                    eftype=es,
                    paired=pair)
            data_df = pd.DataFrame(
                data=data[np.newaxis,:], 
                index=[comparison], 
                columns=data_A.columns)  
        
        ## case not defined
        else:
            lgr.critical(f"comparison '{comparison}' not defined!")
            data_df = None
        
        ## backup and return     
        if store:
            self._groups = groups
            self.comparisons[comparison] = data_df
            if replace:
                lgr.info(f"Replacing 'Y' data with comparison result.")
                self.Y = data_df
                self.y_lab = data_df.index
        else:
            return data_df
        
    # ==============================================================================================

    def predict(self, method, 
                X=None, Y=None, Z=None, 
                comparison=None,
                r_to_z=True, 
                adjust_r2=True, mlr_individual=True,
                store=True, verbose=True, n_proc=None):
        verbose = set_log(verbose)
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.critical("Input data ('X', 'Y') not found. Did you run JuSpyce.fit()?!")
            raise ValueError("No input data!")
        
        # number of runners
        n_proc = self.n_proc if n_proc is None else n_proc
        
        ## overwrite settings from main JuSpyce
        self.r_to_z = r_to_z
        self.adj_r2 = adjust_r2
        self.mlr_individual = mlr_individual
        
        ## get X and Y data (so this function can be run on direct X & Y input data)
        X = self.X if X is None else X
        if comparison is not None:
            Y = self.comparisons[comparison]
            comparison += "-"
        elif Y is not None:
            Y = Y
            comparison = ""
        else:
            Y = self.Y
            comparison = ""
        Z = self.Z if Z is None else Z
        
        # boolean vector to exlude nan parcels
        no_nan = np.array(~self._nan_bool)
        
        ## function to perform prediction target-wise (= per subject), needed for parallelization
        def y_predict(y):    
            
            ## case pearson / case spearman
            if method in ["pearson", "spearman"]:
                rank = True if method=="spearman" else False
                predictions = corr(
                    x=X.values[:,no_nan], # atlas
                    y=Y.iloc[y:y+1,no_nan].values, # subjects
                    correlate="rows", 
                    rank=rank)[-1,:-1]
                if r_to_z:
                    predictions = np.arctanh(predictions)   
                    
            ## case partialpearson / case partialspearman
            elif method in ["partialpearson", "partialspearman"]:
                rank = True if method=="partialspearman" else False
                # iterate x (atlases/predictors)
                predictions = np.zeros(self.X.shape[0], dtype=self._dtype)
                for x in range(self.X.shape[0]):
                    predictions[x] = partialcorr3(
                        x=X.iloc[x,no_nan].values.T, # atlas
                        y=Y.iloc[y,no_nan].values.T, # subject
                        z=Z.values[:,no_nan].T, # data to partial out
                        rank=rank)
                if r_to_z:
                    predictions = np.arctanh(predictions)
                
            ## case slr
            elif method=="slr":
                # iterate x (atlases/predictors)
                predictions = np.zeros(self.X.shape[0], dtype=self._dtype)
                for x in range(self.X.shape[0]):
                    predictions[x] = r2(
                        x=X.iloc[x:x+1,no_nan].values.T, # atlas
                        y=Y.iloc[y:y+1,no_nan].values.T, # subject
                        adj_r2=self.adj_r2)
            ## case mlr
            elif method=="mlr":
                predictions = dict()
                predictions["beta"], predictions["full_r2"] = beta(
                    x=X.values[:,no_nan].T, # atlases
                    y=Y.iloc[y:y+1,no_nan].values.T, # subject      
                    r2=True,
                    adj_r2=self.adj_r2) 
                if mlr_individual:  
                    predictions["individual"] = np.zeros_like(
                        predictions["beta"], dtype=self._dtype)
                    for x in range(self.X.shape[0]):
                        predictions["individual"][x] = r2(
                            x=np.delete(X.values, x, axis=0)[:,no_nan].T, # atlases
                            y=Y.iloc[y:y+1,no_nan].values.T, # subject
                            adj_r2=self.adj_r2)      
                    predictions["individual"] = predictions["full_r2"] - predictions["individual"]
            ## case dominance
            elif method=="dominance":
                predictions = dominance(
                    x=X.values[:,no_nan].T, # atlases
                    y=Y.iloc[y:y+1,no_nan].values.T, # subject   
                    adj_r2=self.adj_r2,
                    verbose=True if verbose=="debug" else False) # dict with dom stats
            ## case not defined
            else:
                lgr.critical(f"Prediction method '{method}' not defined!")
            ## return for collection
            return(predictions)
    
        ## run actual prediction using joblib.Parallel
        predictions_list = Parallel(n_jobs=n_proc)(delayed(y_predict)(y) for y in tqdm(
            range(Y.shape[0]), 
            desc=f"Predicting ({method}, {n_proc} proc)", disable=not verbose))
        
        ## collect data in arrays
        predictions = dict()
        # dominance: dict with one array per dominance stat
        if method=="dominance":
            for dom_stat in ["total", "individual", "relative"]:
                predictions["dominance_"+dom_stat] = np.zeros(
                    (Y.shape[0], X.shape[0]), dtype=self._dtype)
                for y, prediction in enumerate(predictions_list):
                    predictions["dominance_"+dom_stat][y,:] = prediction[dom_stat]
            predictions["dominance_full_r2"] = np.sum(
                predictions["dominance_total"], axis=1)[:,np.newaxis]
        # MLR: dict with one array per stat
        elif method=="mlr":
            predictions["mlr_beta"] = np.zeros(
                (Y.shape[0], X.shape[0]), dtype=self._dtype)
            predictions["mlr_full_r2"] = np.zeros((Y.shape[0],1), dtype=self._dtype)
            if mlr_individual: 
                predictions["mlr_individual"] = np.zeros_like(
                    predictions["mlr_beta"], dtype=self._dtype)
            for y, prediction in enumerate(predictions_list):
                predictions["mlr_beta"][y,:] = prediction["beta"]
                if mlr_individual: predictions["mlr_individual"][y,:] = prediction["individual"]
                predictions["mlr_full_r2"][y] = prediction["full_r2"]
        # all others: one array
        else:
            predictions[method] = np.zeros((Y.shape[0], X.shape[0]), dtype=self._dtype)
            for y, prediction in enumerate(predictions_list):
                predictions[method][y,:] = prediction
        
        ## to dataframe & return
        # return dataframe as attribute of self
        if store:
            if method in ["dominance", "mlr"]:
                for stat in predictions:
                    self.predictions[comparison+stat] = pd.DataFrame(
                        data=predictions[stat], 
                        columns=X.index if not stat.endswith("full_r2") else [stat], 
                        index=Y.index,
                        dtype=self._dtype) 
            else:
                self.predictions[comparison+method] = pd.DataFrame(
                    data=predictions[method],
                    columns=X.index,
                    index=Y.index,
                    dtype=self._dtype) 
        # return numpy array or dict independent of self
        else:
            return predictions
   
    # ==============================================================================================
        
    def permute_maps(self, method, 
                     comparison=None,
                     permute="X", 
                     null_maps=None, use_null_maps=True,
                     null_method="variogram", dist_mat=None, n_perm=1000, 
                     parcellation=None, parc_space=None, parc_hemi=None, centroids=False,
                     r_to_z=None, adjust_r2=None, mlr_individual=None,
                     p_tail=None,
                     n_proc=None, n_proc_predict=1, seed=None,
                     verbose=True, store=True):
        verbose = set_log(verbose)
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.critical("Input data ('X', 'Y') not found. Did you run JuSpyce.fit()?!")
            raise ValueError("No input data!")
        
        ## check correct p_tail
        if p_tail is not None: 
            if not isinstance(p_tail, dict): 
                lgr.critical("If 'p_tail' is defined, it must be a dict mapping (sub-)methods "
                             "to one of ['two', 'upper', 'lower']!")
                
        ## overwrite settings from main JuSpyce
        self.parc = parcellation if parcellation is not None else self.parc
        self.parc_space = parc_space if parc_space is not None else self.parc_space
        self.parc_hemi = parc_hemi if parc_hemi is not None else self.parc_hemi
        adj_r2 = self.adj_r2 if adjust_r2 is None else adjust_r2
        r_to_z = self.r_to_z if r_to_z is None else r_to_z
        mlr_individual = self.mlr_individual if mlr_individual is None else mlr_individual
        n_proc = self.n_proc if n_proc is None else n_proc
        
        ## get "true" prediction
        lgr.info(f"Running 'true' prediction (method = '{method}').")
        prediction_true = self.predict(
            method=method,
            comparison=comparison,
            adjust_r2=adj_r2, 
            r_to_z=r_to_z,
            mlr_individual=mlr_individual,
            store=False,
            verbose=verbose,
            n_proc=n_proc_predict)

        ## generate/ get null maps
        # case null maps given
        if null_maps is not None:
            lgr.info(f"Using provided null maps.")
        # case null maps not given but existing
        elif (null_maps is None) & (use_null_maps==True):
            try:
                null_maps = self.nulls["null_maps"]
            except:
                lgr.info("No null maps found.")
        # case null maps not given & not existing
        if null_maps is None:
            lgr.info(f"Generating null maps for '{permute}' data (n = {n_perm}, "
                     f"null_method = '{null_method}').")
            
            # true data (either X, Y, or comparison Y)
            if comparison is None:
                true_data = self.X if permute=="X" else self.Y
            else:
                true_data = self.comparions[comparison]
            # labels of maps
            map_labs = list(true_data.index)
            
            # case simple permutation
            if (null_method=="random") | (null_method==None):
                # dict to store null maps
                null_maps = dict() 
                # seed
                np.random.seed = seed
                # iterate maps
                for i_map, map_lab in enumerate(tqdm(
                    map_labs, desc=f"Generating {permute} null data", disable=not verbose)):
                    lgr.debug(f"Generating null map for {map_lab}.")
                    # get null data
                    null_maps[map_lab] = np.zeros((n_perm, self.n_parcels), dtype=self._dtype)
                    for i_null in range(n_perm):
                        null_maps[map_lab][i_null,:] = np.random.permutation(
                            true_data.iloc[i_map,:])[:]
                    dist_mat = None
            
            # case variogram -> generate null samples corrected for spatial autocorrelation
            elif null_method=="variogram":
                # null data for all maps 
                null_maps, dist_mat = generate_null_maps(
                    data=true_data, 
                    parcellation=self.parc,
                    parc_space=self.parc_space, 
                    parc_hemi=self.parc_hemi, 
                    parc_density=self.parc_density, 
                    n_nulls=n_perm, 
                    centroids=centroids, 
                    dist_mat=dist_mat,
                    n_cores=n_proc, 
                    seed=seed, verbose=verbose)
            # case not defined
            else:
                lgr.critical(f"Null map generation method '{null_method}' not defined!")
         
        ## define null prediction function for parallelization
        def null_predict(i_null):
             
            # case X nulls
            if permute=="X":
                X = pd.DataFrame(np.c_[[null_maps[m][i_null,:] for m in self.x_lab]])
                null_prediction = self.predict(
                    X=X,
                    Y=self.Y,
                    Z=self.Z,
                    method=method,
                    comparison=comparison,
                    adjust_r2=adj_r2, 
                    r_to_z=r_to_z,
                    mlr_individual=mlr_individual,
                    store=False,
                    verbose=False,
                    n_proc=n_proc_predict)
            # case Y nulls
            else:
                Y = pd.DataFrame(np.c_[[null_maps[m][i_null,:] for m in self.y_lab]])
                null_prediction = self.predict(
                    X=self.X,
                    Y=Y,
                    Z=self.Z,
                    method=method,
                    comparison=comparison,
                    adjust_r2=adj_r2, 
                    r_to_z=r_to_z,
                    mlr_individual=mlr_individual,
                    store=False,
                    verbose=False,
                    n_proc=n_proc_predict)
            # return for collection
            return null_prediction
        
        ## run actual null predictions using joblib.Parallel
        null_predictions_list = Parallel(n_jobs=n_proc)(
            delayed(null_predict)(i_null) for i_null in tqdm(
                range(n_perm), 
                desc=f"Null predictions ({method}, {n_proc} proc)", disable=not verbose))
        # collect data in dict
        null_predictions = dict()
        for i_null, null_prediction in enumerate(null_predictions_list):
            null_predictions[i_null] = null_prediction
        
        ## get p values
        # make method iterable 
        if method in ["dominance", "mlr"]:
            method_i = [k for k in prediction_true if k.startswith(method)] 
        else:
            method_i = [method]
        # define p tails
        if p_tail is None:
            if method=="dominance":
                p_tail = {m:"upper" for m in method_i}
            elif method=="mlr":
                p_tail = {method+"_beta":"two", 
                          method+"_full_r2":"upper", 
                          method+"_individual":"upper"}
            elif method=="slr":
                p_tail = {method:"upper"}
            else:
                p_tail = {method:"two"}
        lgr.info(f"Calculating exact p-values (tails = '{p_tail}').")
        # iterate methods
        p_data = dict()
        for m in method_i:
            p = np.zeros(prediction_true[m].shape, dtype=self._dtype)
            # iterate predictors (columns)
            for x in range(p.shape[1]):
                # iterate targets (rows)
                for y in range(p.shape[0]):
                    true_pred = prediction_true[m][y,x]
                    null_pred = [null_predictions[i][m][y,x] for i in range(n_perm)]
                    # get p value
                    p[y,x] = null_to_p(true_pred, null_pred, tail=p_tail[m])
            # collect data
            p_data[m] = pd.DataFrame(
                data=p,
                columns=self.X.index if "full_r2" not in m else [m],
                index=self.Y.index if comparison is None else self.comparisons[comparison].index,                
                dtype=self._dtype)
            
        ## save & return
        if store:    
            null_data = dict(
                permute=permute,
                n_perm=n_perm,
                null_method=null_method,
                null_maps=null_maps,
                distance_matrix=dist_mat)
            comp = "" if comparison is None else comparison+"-"
            for m in p_data:
                self.p_predictions[comp+m] = p_data[m]
            for k in null_data: 
                self.nulls[k] = null_data[k]
            self.nulls["predictions-"+comp+method] = null_predictions
        return p_data, null_predictions    

    # ==============================================================================================

    def permute_groups(self, method, comparison, groups,
                       n_perm=1000, 
                       p_tail="two",
                       r_to_z=None, adjust_r2=None, mlr_individual=None,
                       n_proc=None, n_proc_predict=1, seed=None,
                       verbose=True, store=True):
        verbose = set_log(verbose)
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.critical("Input data ('X', 'Y') not found. Did you run JuSpyce.fit()?!")
            raise ValueError("No input data!")
        
        ## overwrite settings from init
        adj_r2 = self.adj_r2 if adjust_r2 is None else adjust_r2
        r_to_z = self.r_to_z if r_to_z is None else r_to_z
        mlr_individual = self.mlr_individual if mlr_individual is None else mlr_individual
        n_proc = self.n_proc if n_proc is None else n_proc
        
        ## get "true" comparison and prediction
        lgr.info(f"Running 'true' group comparison and prediction "
                 f"(comparison = '{comparison}', method = '{method}').")
        # comparison
        Yc_true = self.compare(
            comparison=comparison, 
            groups=groups, 
            store=False, verbose=verbose)
        # prediction
        prediction_true = self.predict(
            X=self.X,
            Y=Yc_true,
            Z=self.Z,
            method=method,
            adjust_r2=adj_r2, 
            r_to_z=r_to_z,
            mlr_individual=mlr_individual,
            store=False,
            verbose=verbose,
            n_proc=n_proc_predict)
        
        ## prepare null comparisons/predictions
        # get list of permuted group labels
        np.random.seed(seed)
        groups_null = [np.random.permutation(groups) for _ in range(n_perm)]
        # null comparison function
        def null_compare_predict(null_groups):
            # compare
            Yt_null = self.compare(
                comparison=comparison, 
                groups=null_groups, 
                store=False, verbose=False)
            # prediction
            prediction_null = self.predict(
                X=self.X,
                Y=Yt_null,
                Z=self.Z,
                method=method,
                adjust_r2=adj_r2, 
                r_to_z=r_to_z,
                mlr_individual=mlr_individual,
                store=False, verbose=False,
                n_proc=n_proc_predict)
            return prediction_null
        
        ## run null comparisons/predictions in parallel 
        lgr.info(f"Running null comparisons and predictions (comparison = '{comparison}', "
                 f"method = '{method}').")
        null_predictions_list = Parallel(n_jobs=n_proc)(
            delayed(null_compare_predict)(g) for g in tqdm(
                groups_null, 
                desc=f"Null comparisons ({method}, {n_proc} proc)", disable=not verbose))
        # collect null data in dict
        null_predictions = dict()
        for i_null, null_prediction in enumerate(null_predictions_list):
            null_predictions[i_null] = null_prediction
        
        ## get p values from null distributions
        # make method iterable and define p-tails:
        if method=="dominance":
            method_i = [k for k in prediction_true if k.startswith("dominance")] 
        elif method=="mlr":
            method_i = [k for k in prediction_true if k.startswith("mlr")] 
        else:
            method_i = [method]
        lgr.info(f"Calculating exact p-values (tails = '{p_tail}').")
        # iterate methods
        p_data = dict()
        for m in method_i:
            p = np.zeros_like(prediction_true[m])
            # iterate predictors (columns)
            for x in range(p.shape[1]):
                # iterate targets (rows)
                for y in range(p.shape[0]):
                    true_pred = prediction_true[m][y,x]
                    null_pred = [null_predictions[i][m][y,x] for i in range(n_perm)]
                    # get p value
                    p[y,x] = null_to_p(true_pred, null_pred, tail=p_tail)
            # collect data
            p_data[m] = pd.DataFrame(
                data=p,
                columns=self.X.index if "full_r2" not in m else [m],
                index=Yc_true.index,
                dtype=self._dtype)
            
        ## save & return
        if store:    
            for m in p_data:
                self.comparisons["Y-"+comparison] = prediction_true
                self.p_comparisons[comparison+"-"+m] = p_data[m]
            self.nulls["comparisons-"+comparison+"-"+method] = null_predictions
        return p_data, prediction_true, null_predictions    
        
    # ==============================================================================================

    def correct_p(self, analysis="predictions", method="all",
                  mc_alpha=0.05, mc_method="fdr_bh", mc_dimension="array", store=True):
        
        # get p data depending on analysis type
        if analysis=="predictions":
            p_value_dict = self.p_predictions
        elif analysis=="comparisons":
            p_value_dict = self.p_comparisons
        # get list of available data depending on method
        method = [method] if isinstance(method, str) else method
        if (method==["all"]) | (method==[True]):
            # list of all p-value dataframes, reduce to unique uncorrected p-values
            method = list(set([key.split("--")[0] for key in p_value_dict.keys()]))
        # get dimension of array to correct along
        if mc_dimension in ["x", "X", "c", "col", "cols", "column", "columns"]:
            how = "c"
        elif mc_dimension in ["y", "Y", "r", "row", "rows"]:
            how = "r"
        else:
            how = "a"
        
        # get p values
        p_corr = dict()
        for m in method:
            p_corr[m+"--"+mc_method], _ = mc_correction(
                p_value_dict[m], 
                alpha=mc_alpha, 
                method=mc_method, 
                how=how, 
                dtype=self._dtype)
        # save and return
        if store:
            if analysis=="predictions":
                for key in p_corr: self.p_predictions[key] = p_corr[key]
            elif analysis=="comparisons":
                for key in p_corr: self.p_comparisons[key] = p_corr[key]
        return p_corr
              
    # ==============================================================================================

    def to_pickle(self, filepath, verbose=True):
        set_log(verbose)
        
        ext = os.path.splitext(filepath)[1]
        # compressed
        if ext==".gz":
            with gzip.open(filepath, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if verbose: 
                lgr.info(f"Saved complete gzip compressed object to {filepath}.")
        # uncompressed
        elif ext in [".pkl", ".pickle"]:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if verbose: 
                lgr.info(f"Saved complete uncompressed object to {filepath}.")
        else:
            lgr.critical(f"Filetype *{ext} not known. Choose one of: '.pbz2', '.pickle', '.pkl'.")     

    # ==============================================================================================

    def copy(self, deep=True, verbose=True):
        set_log(verbose)
        
        lgr.info(f"Creating{' deep ' if deep else ' '}copy of JuSpyce object.")
        if deep==True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)        
            
    # ==============================================================================================

    @staticmethod 
    def from_pickle(filepath, verbose=True):
        set_log(verbose)
        
        ext = os.path.splitext(filepath)[1]
        # compressed
        if ext==".gz":
            with gzip.open(filepath, "rb") as f:
                juspyce_object = pickle.load(f)
            lgr.info(f"Loaded complete object from {filepath}.")
        # uncompressed
        elif ext in [".pkl", ".pickle"]:
            with open(filepath, "rb") as f:
                juspyce_object = pickle.load(f)
            lgr.info(f"Loaded complete object from {filepath}.")
        else:
            lgr.critical(f"Filetype *{ext} not known. Choose one of: '.pbz2', '.pickle', '.pkl'.")     
        # return
        return juspyce_object
