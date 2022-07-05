
import copy
import gzip
import logging
import os
import pickle
import sys

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
lgr.setLevel(logging.INFO)
            
class JuSpyce:
    """Class JuSpyce
    """   

    def __init__(self, x, y, z=None, parcellation=None, data_space="MNI152",
                 x_labels=None, y_labels=None, z_labels=None,
                 standardize="xy", 
                 parcellation_labels=None, parcellation_space="MNI152", 
                 parcellation_hemi=["L", "R"], parcellation_density="10k",
                 drop_nan=False,
                 n_proc=1, dtype=np.float32,
                 ):
        
        self.x = x
        self.y = y
        self.z = z
        self.x_lab = x_labels
        self.y_lab = y_labels
        self.z_lab = z_labels
        self.data_space = data_space
        self.parc = parcellation
        self.parc_lab = parcellation_labels
        self.parc_space = parcellation_space
        self.parc_hemi = parcellation_hemi
        self.parc_density = parcellation_density
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
        # empty data storage
        self.predictions = dict()
        self.nulls = dict()
        self.group_comparisons = dict()
        self.p_predictions = dict()
        self.p_group_comparisons = dict()
    
    # ==============================================================================================
    
    def fit(self):
        
        ## extract input data
        # predictors -> usually e.g. PET atlases
        lgr.info("Checking input data for x (should be, e.g., PET data):")
        self.X = get_input_data(self.x, 
                                data_labels=self.x_lab,
                                data_space=self.data_space, 
                                parcellation=self.parc, 
                                parc_labels=self.parc_lab,
                                parc_hemi=self.parc_hemi,
                                parc_space=self.parc_space)
        self.n_predictors = self.X.shape[0]
        lgr.info(f"Got 'x' data for {self.n_predictors} x {self.X.shape[1]} parcels.")
        # targets -> usually e.g. subject data or group-level outcome data
        lgr.info("Checking input data for y (should be, e.g., subject data):")
        self.Y = get_input_data(self.y, 
                                data_labels=self.y_lab,
                                data_space=self.data_space, 
                                parcellation=self.parc, 
                                parc_labels=self.parc_lab,
                                parc_hemi=self.parc_hemi,
                                parc_space=self.parc_space)
        self.n_targets = self.Y.shape[0]
        lgr.info(f"Got 'y' data for {self.n_targets} x {self.Y.shape[1]} parcels.")
        # data to control correlations for
        if self.z is not None:
            lgr.info("Checking input data for z (should be, e.g., grey matter data):")
            if isinstance(self.z, str):
                if self.z in ["GM", "GMV", "gm", "gmv"]:
                    lgr.info("Using nilearn grey matter template as 'z' to control for GMV.")
                    self.z = [get_template("gm", resolution=1)]
                    self.z_lab = ["gm"]
            self.Z = get_input_data(self.z, 
                                    data_labels=self.z_lab,
                                    data_space=self.data_space, 
                                    parcellation=self.parc, 
                                    parc_labels=self.parc_lab,
                                    parc_hemi=self.parc_hemi,
                                    parc_space=self.parc_space)
            lgr.info(f"Got 'z' data for {self.Z.shape[0]} x {self.Z.shape[1]} parcels.")
        else:
            self.Z = None
        
        ## check parcel number
        if self.X.shape[1]!=self.Y.shape[1]:
            lgr.error("Got differing numbers of parcels in 'x' & 'y' data!")
        if self.Z is not None:
            if self.X.shape[1]!=self.Z.shape[1]:
                lgr.error("Got differing numbers of parcels in 'z' data!")
                  
        ## deal with nan's
        self._nan_bool = pd.concat([self.X, self.Y, self.Z], axis=0).isnull().any(axis=0)
        # case remove nan parcels completely
        if self._drop_nan==True:
            lgr.warning(f"Dropping {np.sum(self._nan_bool)} parcels with nan's. This will lead to problems with null map generation!")
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
    
    def transform(self, transform, dataset, groups=None, 
                  n_components=None, min_ev=None, fa_method="minres", fa_rotation="promax",
                  seed=None, store=True, verbose=True):
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.error("Input data ('X', 'Y') not found. Did you run CMC.fit()?!")
        
        ## backup data to be changed by transforms
        self._transform_count += 1
        def backup(data):
            if not hasattr(self, "_transform_backup"): 
                self._transform_backup = dict()
            self._transform_backup[f"{self._transform_count}_{transform}_{dataset}"] = data 
        
        ## get data
        data_orig = self.Y if dataset=="Y" else self.X
        data_orig_nan = data_orig.loc[:, ~self._nan_bool]
        
        ## case mean
        if transform=="mean":
            lgr.info(f"Calculating parcel-wise mean of {dataset}.")
            data = np.nanmean(data_orig.values, axis=0)
            data_df = pd.DataFrame(data=data[np.newaxis,:], index=[transform], columns=data_orig.columns, dtype=self._dtype)   
        
        ## case partial
        elif transform=="partial":
            lgr.info(f"Regressing 'Z' from '{dataset}': new {dataset} = residuals.")
            
            # get residuals
            data = np.zeros(data_orig_nan.shape, dtype=self._dtype)
            for xy in tqdm(range(data.shape[0]), disable=not verbose):
                data[xy,:] = residuals(x=self.Z.values[:,~self._nan_bool].T, 
                                       y=data_orig_nan.iloc[xy,:].values.T)
            # save
            data_df = fill_nan(pd.DataFrame(data=data, index=data_orig_nan.index, 
                                            columns=data_orig_nan.columns),
                               self._nan_cols, self._nan_labs, "col")
        
        ## case pca / case ICA
        elif transform in ["pca", "ica", "fa"]:
            lgr.info(f"Calculating {transform} on '{dataset}' data.")
            data, ev, loadings = reduce_dimensions(data=data_orig_nan.T, 
                                                   method=transform, 
                                                   n_components=n_components, 
                                                   min_ev=min_ev,
                                                   fa_method=fa_method, 
                                                   fa_rotation=fa_rotation,
                                                   seed=seed)
            # save
            data_df = fill_nan(pd.DataFrame(data=data.T, index=[f"c{i}" for i in range(data.shape[1])], 
                                            columns=data_orig_nan.columns, dtype=self._dtype),
                               self._nan_cols, self._nan_labs, "col")
            loadings = pd.DataFrame(loadings, columns=data_df.index, index=data_orig.index, dtype=self._dtype)
        
        ## case A-mean(B) / case mean(A)-mean(B) 
        ## case cohen(A,B) / case hedge(A,B) / case pairedcohen(A,B)
        elif transform in ["A-mean(B)", "mean(A)-mean(B)", "cohen(A,B)", "hedge(A,B)", "pairedcohen(A,B)"]:
            # group variable
            if groups is None:
                lgr.error("For transform '{transform}' you must provide a grouping variable!")
            groups = np.array(groups)
            idc = np.sort(np.unique(groups))
            if len(idc) > 2:
                lgr.error("Function not defined for > 2 grouping categories!", idc)
            if len(groups) != data_orig.shape[0]:
                lgr.error(f"Length of 'groups' ({len(groups)}) does not match length of {dataset} ({data_orig.shape[0]})!")
            # group dfs
            data_A = data_orig[groups==idc[0]]
            data_B = data_orig[groups==idc[1]]   
                  
            # transform            
            ## case A-mean(B)
            if transform=="A-mean(B)":
                lgr.info(f"Subtracting parcel-wise mean of B from A: new {dataset} = {dataset}[A] - mean({dataset}[B]).")
                data = data_A - np.nanmean(data_B, axis=0)
                data_df = pd.DataFrame(data=data, index=data_A.index, columns=data_A.columns, dtype=self._dtype)  

            ## case mean(A)-mean(B)
            elif transform=="mean(A)-mean(B)":
                lgr.info(f"Subtracting parcel-wise mean of B from mean of A: new {dataset} = mean({dataset}[A]) - mean({dataset}[B]).")
                data = np.nanmean(data_A, axis=0) - np.nanmean(data_B, axis=0)
                data_df = pd.DataFrame(data=data[np.newaxis,:], index=[transform], columns=data_A.columns, dtype=self._dtype)
            
            # case cohen(A,B) / case hedge(A,B) / case pairedcohen(A,B)
            else:
                es = "cohen" if "cohen" in transform else "hedges"
                pair = True if "paired" in transform else False
                lgr.info(f"Calculating parcel-wise effect size between A and B ({es}, paired: {pair}).")
                data = np.zeros(self.n_parcels, dtype=self._dtype)
                for p in tqdm(range(self.n_parcels), disable=not verbose):
                    data[p] = compute_effsize(x=data_A.iloc[:,p].values,
                                              y=data_B.iloc[:,p].values,
                                              eftype=es,
                                              paired=pair)
                data_df = pd.DataFrame(data=data[np.newaxis,:], index=[transform], columns=data_A.columns)  
        
        ## case not defined
        else:
            lgr.error(f"transform '{transform}' not defined!")
            data_df = data_orig
            data_orig = None
        
        ## backup and return     
        if store:
            backup(data_orig) 
            if dataset=="Y":
                self.Y = data_df
                self.n_targets = data_df.shape[0]
                self.y_lab = data_df.index
            else:
                self.X = data_df
                self.n_predictors = data_df.shape[0]
                self.x_lab = data_df.index
            self.groups = groups
            if transform in ["pca", "ica", "fa"]:
                self.dim_red = dict(ev=ev,
                                    loadings=loadings)
        return data_df
        
    # ==============================================================================================

    def predict(self, method, adjust_r2=True, r_to_z=True, 
                X=None, Y=None, Z=None, 
                store=True, verbose=True, n_proc=None):
        
        ## check if fit was run
        if not (hasattr(self, "X") | hasattr(self, "Y")):
            lgr.error("Input data ('X', 'Y') not found. Did you run CMC.fit()?!")
            sys.exit()
        
        # number of runners
        n_proc = self.n_proc if n_proc is None else n_proc
        
        ## overwrite settings from main CMC
        self.r_to_z = r_to_z
        self.adj_r2 = adjust_r2
        
        ## get X and Y data (so this function can be run on direct X & Y input data)
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        Z = self.Z if Z is None else Z
        
        # array to store results
        #predictions = np.zeros((self.n_targets, self.n_predictors))
        
        # boolean vector to exlude nan parcels
        no_nan = np.array(~self._nan_bool)
        
        ## function to perform prediction target-wise (= per subject), needed for parallelization
        def y_predict(y):    
            
            ## case pearson / case spearman
            if method in ["pearson", "spearman"]:
                rank = True if method=="spearman" else False
                predictions = corr(x=X.values[:,no_nan], # atlas
                                   y=Y.iloc[y:y+1,no_nan].values, # subjects
                                   correlate="rows", 
                                   rank=rank)[-1,:-1]
                if r_to_z:
                    predictions = np.arctanh(predictions)   
                    
            ## case partialpearson / case partialspearman
            elif method in ["partialpearson", "partialspearman"]:
                rank = True if method=="partialspearman" else False
                # iterate x (atlases/predictors)
                predictions = np.zeros(self.n_predictors, dtype=self._dtype)
                for x in range(self.n_predictors):
                    predictions[x] = partialcorr3(x=X.iloc[x,no_nan].values.T, # atlas
                                                  y=Y.iloc[y,no_nan].values.T, # subject
                                                  z=Z.values[:,no_nan].T, # data to partial out
                                                  rank=rank)
                if r_to_z:
                    predictions = np.arctanh(predictions)
                
            ## case slr
            elif method=="slr":
                # iterate x (atlases/predictors)
                predictions = np.zeros(self.n_predictors, dtype=self._dtype)
                for x in range(self.n_predictors):
                    predictions[x] = r2(x=X.iloc[x:x+1,no_nan].values.T, # atlas
                                        y=Y.iloc[y:y+1,no_nan].values.T, # subject
                                        adj_r2=self.adj_r2)
            ## case mlr
            elif method=="mlr":
                predictions = dict()
                predictions["beta"], predictions["full_r2"] = beta(x=X.values[:,no_nan].T, # atlases
                                                                   y=Y.iloc[y:y+1,no_nan].values.T, # subject      
                                                                   r2=True,
                                                                   adj_r2=self.adj_r2)                      
            ## case dominance
            elif method=="dominance":
                predictions = dominance(x=X.values[:,no_nan].T, # atlases
                                        y=Y.iloc[y:y+1,no_nan].values.T, # subject   
                                        adj_r2=self.adj_r2,
                                        verbose=True if verbose=="debug" else False) # dict with dom stats
            ## case not defined
            else:
                lgr.error(f"Prediction method '{method}' not defined!")
            ## return for collection
            return(predictions)
    
        ## run actual prediction using joblib.Parallel
        predictions_list = Parallel(n_jobs=n_proc)(delayed(y_predict)(y) for y in tqdm(
            range(self.n_targets), desc=f"Predicting ({method}, {n_proc} proc)", disable=not verbose))
        
        ## collect data in arrays
        predictions = dict()
        # dominance: dict with one array per dominance stat
        if method=="dominance":
            for dom_stat in ["total", "individual", "relative"]:
                predictions["dominance_"+dom_stat] = np.zeros((self.n_targets, self.n_predictors), dtype=self._dtype)
                for y, prediction in enumerate(predictions_list):
                    predictions["dominance_"+dom_stat][y,:] = prediction[dom_stat]
            predictions["dominance_full_r2"] = np.sum(predictions["dominance_total"], axis=1)[:,np.newaxis]
        # MLR: dict with one array per stat
        elif method=="mlr":
            predictions["mlr_beta"] = np.zeros((self.n_targets, self.n_predictors), dtype=self._dtype)
            predictions["mlr_full_r2"] = np.zeros((self.n_targets,1), dtype=self._dtype)
            for y, prediction in enumerate(predictions_list):
                predictions["mlr_beta"][y,:] = prediction["beta"]
                predictions["mlr_full_r2"][y] = prediction["full_r2"]
        # all others: one array
        else:
            predictions[method] = np.zeros((self.n_targets, self.n_predictors), dtype=self._dtype)
            for y, prediction in enumerate(predictions_list):
                predictions[method][y,:] = prediction
        
        ## to dataframe & return
        # return dataframe as attribute of self
        if store:
            if method in ["dominance", "mlr"]:
                for stat in predictions:
                    self.predictions[stat] =  pd.DataFrame(
                        data=predictions[stat], 
                        columns=self.x_lab if not stat.endswith("full_r2") else [stat], 
                        index=self.y_lab,
                        dtype=self._dtype) 
            else:
                self.predictions[method] = pd.DataFrame(
                    data=predictions[method],
                    columns=self.x_lab,
                    index=self.y_lab,
                    dtype=self._dtype) 
        # return numpy array or dict independent of self
        else:
            return predictions
   
    # ==============================================================================================
        
    def permute_maps(self, method, permute="X", null_maps=None, use_null_maps=True,
                     null_method="variogram", dist_mat=None, n_perm=1000, 
                     adjust_r2=None, r_to_z=None,
                     p_tail=None,
                     n_proc=None, n_proc_predict=1, seed=None,
                     parcellation=None, parc_space=None, parc_hemi=None, centroids=False,
                     verbose=True, store=True):
        
        ## check if predict was run
        if method not in self.predictions:
            if not [p for p in self.predictions if p.startswith(method)]:
                lgr.error(f"Data for prediction method '{method}' not found. Did you run CMC.predict(method='{method}', store=True)?!")
                sys.exit()
        
        ## check correct p_tail
        if p_tail is not None: 
            if ~isinstance(p_tail, dict): 
                lgr.error("If 'p_tail' is defined, it must be a dict mapping (sub-)method to 'p_tails'!")
                sys.exit()
                
        ## overwrite settings from main CMC
        self.parc = parcellation if parcellation is not None else self.parc
        self.parc_space = parc_space if parc_space is not None else self.parc_space
        self.parc_hemi = parc_hemi if parc_hemi is not None else self.parc_hemi
        adj_r2 = self.adj_r2 if adjust_r2 is None else adjust_r2
        r_to_z = self.r_to_z if r_to_z is None else r_to_z
        n_proc = self.n_proc if n_proc is None else n_proc

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
            lgr.info(f"Generating null maps for '{permute}' data (n = {n_perm}, null_method = '{null_method}', method = '{method}', n_prc = {n_proc}).")
            
            # true data (either X or Y)
            true_data = self.X if permute=="X" else self.Y
            # labels of maps
            map_labs = list(true_data.index)
            
            # case simple permutation
            if (null_method=="random") | (null_method==None):
                # dict to store null maps
                null_maps = dict() 
                # seed
                np.random.seed = seed
                # iterate maps
                for i_map, map_lab in enumerate(tqdm(map_labs, desc=f"Generating {permute} null data", disable=not verbose)):
                    lgr.debug(f"Generating null map for {map_lab}.")
                    # get null data
                    null_maps[map_lab] = np.zeros((n_perm, self.n_parcels), dtype=self._dtype)
                    for i_null in range(n_perm):
                        null_maps[map_lab][i_null,:] = np.random.permutation(true_data.iloc[i_map,:])[:]
                    dist_mat = None
            
            # case variogram -> generate null samples corrected for spatial autocorrelation
            elif null_method=="variogram":
                # null data for all maps 
                null_maps, dist_mat = generate_null_maps(data=true_data, 
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
                lgr.error(f"Null map generation method '{null_method}' not defined!")
         
        ## define null prediction function for parallelization
        def null_predict(i_null):
             
            # case X nulls
            if permute=="X":
                X = pd.DataFrame(np.c_[[null_maps[m][i_null,:] for m in self.x_lab]])
                null_prediction = self.predict(X=X,
                                               Y=self.Y,
                                               Z=self.Z,
                                               method=method,
                                               adjust_r2=adj_r2, 
                                               r_to_z=r_to_z,
                                               store=False,
                                               verbose=False,
                                               n_proc=n_proc_predict)
            # case Y nulls
            else:
                Y = pd.DataFrame(np.c_[[null_maps[m][i_null,:] for m in self.y_lab]])
                null_prediction = self.predict(X=self.X,
                                               Y=Y,
                                               Z=self.Z,
                                               method=method,
                                               adjust_r2=adj_r2, 
                                               r_to_z=r_to_z,
                                               store=False,
                                               verbose=False,
                                               n_proc=n_proc_predict)
            # return for collection
            return null_prediction
        
        ## run actual null predictions using joblib.Parallel
        null_predictions_list = Parallel(n_jobs=n_proc)(delayed(null_predict)(i_null) for i_null in tqdm(
            range(n_perm), desc=f"Null predictions ({method}, {n_proc} proc)", disable=not verbose))
        # collect data in array
        null_predictions = dict()
        for i_null, null_prediction in enumerate(null_predictions_list):
            null_predictions[i_null] = null_prediction
        
        ## get p values
        # make method iterable and define p-tails:
        if method=="dominance":
            method_i = ["dominance_"+c for c in ["total", "individual", "relative", "full_r2"]] 
            p_tail = {m:"upper" for m in method_i} if p_tail is None else p_tail
        elif method=="mlr":
            method_i = ["mlr_"+c for c in ["beta", "full_r2"]]
            p_tail = {"mlr_beta":"two", "mlr_full_r2":"upper"} if p_tail is None else p_tail
        else:
            method_i = [method]
            p_tail = {method:"upper"} if method=="slr" else {method:"two"} if p_tail is None else p_tail
        lgr.info(f"Calculating exact p-values (tails = {p_tail}).")
        # iterate methods
        p_data = dict()
        for m in method_i:
            p = np.zeros(self.predictions[m].shape, dtype=self._dtype)
            # iterate predictors (columns)
            for x in range(p.shape[1]):
                # iterate targets (rows)
                for y in range(p.shape[0]):
                    true_pred = self.predictions[m].iloc[y,x]
                    null_pred = [null_predictions[i][m][y,x] for i in range(n_perm)]
                    # get p value
                    p[y,x] = null_to_p(true_pred, null_pred, tail=p_tail[m])
            # collect data
            p_data[m] = pd.DataFrame(data=p,
                                     columns=self.predictions[m].columns,
                                     index=self.predictions[m].index,
                                     dtype=self._dtype)
            
        ## save & return
        if store:    
            null_data = dict(
                permute=permute,
                n_perm=n_perm,
                null_method=null_method,
                null_maps=null_maps,
                distance_matrix=dist_mat)
            for m in p_data:
                self.p_predictions[m] = p_data[m]
            for k in null_data: 
                self.nulls[k] = null_data[k]
            self.nulls[method+"-predictions"] = null_predictions
        return p_data, null_predictions    

    # ==============================================================================================

    def correct_p(self, analysis="predictions", method="all",
                  mc_alpha=0.05, mc_method="fdr_bh", mc_dimension="array", store=True):
        
        # get p data depending on analysis type
        if analysis=="predictions":
            p_value_dict = self.p_predictions
        elif analysis=="group_comparisons":
            p_value_dict = self.p_group_comparisons
        # get list of available data depending on method
        method = [method] if isinstance(method, str) else method
        if (method==["all"]) | (method==[True]):
            # list of all p-value dataframes, reduce to unique uncorrected p-values
            method = list(set([key.split("-")[0] for key in p_value_dict.keys()]))
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
            p_corr[m+"-"+mc_method], _ = mc_correction(p_value_dict[m], alpha=mc_alpha, 
                                                       method=mc_method, how=how, dtype=self._dtype)
        # save and return
        if store:
            if analysis=="predictions":
                for key in p_corr: self.p_predictions[key] = p_corr[key]
            elif analysis=="group_comparisons":
                for key in p_corr: self.p_group_comparisons[key] = p_corr[key]
        return p_corr
        
    # ==============================================================================================

    def permute_groups():
        pass
        
    # ==============================================================================================

    def to_pickle(self, filepath, verbose=True):
        
        ext = os.path.splitext(filepath)[1]
        # compressed
        if ext==".gz":
            with gzip.open(filepath, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if verbose: lgr.info(f"Saved complete gzip compressed object to {filepath}.")
        # uncompressed
        elif ext in [".pkl", ".pickle"]:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if verbose: lgr.info(f"Saved complete uncompressed object to {filepath}.")
        else:
            lgr.error(f"Filetype *{ext} not known. Choose one of: '.pbz2', '.pickle', '.pkl'.")     

    # ==============================================================================================

    def copy(self, deep=True, verbose=True):
        
        if verbose: lgr.info(f"Creating{' deep ' if deep else ' '}copy of JuSpyce object.")
        if deep==True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)        
            
    # ==============================================================================================

    @staticmethod 
    def from_pickle(filepath, verbose=True):
        
        ext = os.path.splitext(filepath)[1]
        # compressed
        if ext==".gz":
            with gzip.open(filepath, "rb") as f:
                cmc = pickle.load(f)
            if verbose: lgr.info(f"Loaded complete object from {filepath}.")
        # uncompressed
        elif ext in [".pkl", ".pickle"]:
            with open(filepath, "rb") as f:
                cmc = pickle.load(f)
            if verbose: lgr.info(f"Loaded complete object from {filepath}.")
        else:
            lgr.error(f"Filetype *{ext} not known. Choose one of: '.pbz2', '.pickle', '.pkl'.")     
        # return
        return cmc
