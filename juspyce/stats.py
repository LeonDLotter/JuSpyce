import logging
from itertools import combinations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy.stats import rankdata, zscore
from sklearn.decomposition import PCA, FastICA
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
lgr.setLevel(logging.INFO)


def corr(x, y, correlate="rows", rank=True):
    
    ## orientation
    if correlate.startswith("r"):
        axis = 0
        rowvar = True
    elif correlate.startswith("c"):
        axis = 1
        rowvar = False
    else:
        print(f"Option correlate=='{correlate}' not defined!")  
    
    ## calculate
    # two 1D vectors
    if (len(x.shape)==1) & (len(y.shape)==1):
        if rank==True:
            x = rankdata(x)
            y = rankdata(y)
        r = np.corrcoef(x, y)[1,0]
    # at least one 2D array
    elif (len(x.shape)==2) & (len(y.shape)==2):
        if rank==True:
            x = rankdata(x, axis=rowvar)
            y = rankdata(y, axis=rowvar)
        r = np.corrcoef(np.append(x, y, axis=axis), rowvar=rowvar)
    # to many dimensions
    else:
        print("Inputs must be two 1D arrays or two 2D arrays!")
        r = np.nan
    
    ## return
    return r


def partialcorr3(x, y, z, rank=False):
    """Computes partial correlation between {x} and {y} controlled for {z}

    Args:
        x (array-like): input vector 1
        y (array-like): input vector 2
        z (array-like): input vector to be controlled for
        rank (bool, optional): True or False. Defaults to False. -> Pearson correlation

    Returns:
        rp (float): (ranked) partial correlation coefficient between x and y
    """    

    C = np.column_stack((x, y, z))
    
    if rank:
        C = rankdata(C, axis=0)
        
    corr = np.corrcoef(C, rowvar=False) # Pearson product-moment correlation coefficients.
    corr_inv = np.linalg.inv(corr) # the (multiplicative) inverse of a matrix.
    rp = -corr_inv[0,1] / (np.sqrt(corr_inv[0,0] * corr_inv[1,1]))
    
    return rp


def r2(x, y, adj_r2=True):
    """Compute R2 for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        adj_r2 (bool, optional): Calculate adjusted R2. Defaults to True.

    Returns:
        float: (adjusted) R2
    """
    
    X = np.c_[x, np.ones(x.shape[0])] 
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    ss_res = np.sum((y-y_hat)**2)       
    ss_tot = np.sum((y-np.mean(y))**2)   
    r2 = 1 - ss_res / ss_tot  
    if adj_r2:
        return 1 - (1-r2) * (len(y)-1) / (len(y)-x.shape[1]-1)
    else:
        return r2
    
    
def beta(x, y, r2=False, adj_r2=True):
    """Compute beta coefficients for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)

    Returns:
        numpy.ndarray: 1D array of beta coefficients (w/o intercept)
    """
    
    X = np.c_[x, np.ones(x.shape[0])] 
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    if r2==False:
        return beta[:-1].flatten()
    else:
        y_hat = np.dot(X, beta)
        ss_res = np.sum((y-y_hat)**2)       
        ss_tot = np.sum((y-np.mean(y))**2)   
        r2 = 1 - ss_res / ss_tot  
        if adj_r2:
            r2a = 1 - (1-r2) * (len(y)-1) / (len(y)-x.shape[1]-1)
            return beta[:-1].flatten(), r2a
        else:
            return beta[:-1].flatten(), r2
    

def residuals(x, y):
    """Compute residuals for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)

    Returns:
        numpy.ndarray: 1D array of residuals
    """
    X = np.c_[x, np.ones(x.shape[0])] 
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    return y - y_hat


def dominance(x, y, adj_r2=False, verbose=True):
    
    def del_from_tuple(tpl, elem):
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)

    if verbose: lgr.info(f"Running dominance analysis with {x.shape[1]} "
                         f"predictors and {len(y)} features.")
    
    ## print total rsquare
    rsq_total = r2(x=x, y=y, adj_r2=adj_r2)
    if verbose: lgr.info(f"Full model R^2 = {rsq_total:.03f}")
    dom_stats = dict()
    dom_stats["full_rsq"] = rsq_total
    
    ## get possible predictor combinations
    n_pred = x.shape[1]
    pred_combs = [list(combinations(range(n_pred), i)) for i in range(1, n_pred+1)]
    
    ## calculate R2s
    if verbose: lgr.info("Calculating models...")
    rsqs = dict()
    for len_group in tqdm(pred_combs, desc='Iterating over len groups', disable=not verbose):
        for pred_idc in tqdm(len_group, desc='Inside loop', disable=True):
            rsq = r2(x=x[:, pred_idc], y=y, adj_r2=adj_r2)
            rsqs[pred_idc] = rsq

    ## collect metrics
    # individual dominance
    if verbose: lgr.info("Calculating individual dominance.")
    dom_stats["individual"] = np.zeros((n_pred))    
    for i in range(n_pred):
        dom_stats["individual"][i] = rsqs[(i,)]
    dom_stats["individual"] = dom_stats["individual"].reshape(1, -1)
        
    # partial dominance
    if verbose: lgr.info("Calculating partial dominance.")
    dom_stats["partial"] = np.zeros((n_pred, n_pred-1)) 
    for i in range(n_pred - 1):
        i_len_combs = list(combinations(range(n_pred), i + 2))
        for j_node in range(n_pred):
            j_node_sel = [v for v in i_len_combs if j_node in v]
            reduced_list = [del_from_tuple(comb, j_node) for comb in j_node_sel]
            diff_values = [rsqs[j_node_sel[i]] - rsqs[reduced_list[i]] for i in range(
                len(reduced_list))]
            dom_stats["partial"][j_node,i] = np.mean(diff_values)
    #dom_stats["partial"] = dom_stats["partial"].mean(axis=1)

    # total dominance
    if verbose: lgr.info("Calculating total dominance.")
    dom_stats["total"] = np.mean(np.c_[dom_stats["individual"].T, dom_stats["partial"]], axis=1)
        
    # relative contribution
    dom_stats["relative"] = dom_stats["total"] / rsq_total
    
    ## sanity check
    if not np.allclose(np.sum(dom_stats["total"]), rsq_total):
        lgr.error(f"Sum of total dominance ({np.sum(dom_stats['total'])}) does not "
                  f"equal full model R^2 ({rsq_total})! ")
    
    return dom_stats


def zscore_df(df, along="cols"):
    """Z-standardizes array and returns pandas dataframe.

    Args:
        df (pandas dataframe): input dataframe
        along (str, optional): Either "cols" or "rows". Defaults to "cols".

    Returns:
        pd.DataFrame or pd.Series: standardized dataframe/series
    """    
    
    if along=="cols":
        axis = 0
    elif along=="rows":
        axis = 1
    else:
        print(f"Option along=={along} not defined!")
    
    if isinstance(df, pd.DataFrame):
        df_scaled = pd.DataFrame(
            data=zscore(df, axis=axis, nan_policy="omit"),
            columns=df.columns,
            index=df.index
        )
    elif isinstance(df, pd.Series):
        df_scaled = pd.DataFrame(
            data=zscore(df, axis=axis, nan_policy="omit"),
            columns=df.name,
            index=df.index
        )
    elif isinstance(df, np.ndarray):
        df_scaled = pd.DataFrame(
            data=zscore(df, axis=axis, nan_policy="omit")
        )
    
    return df_scaled


def reduce_dimensions(data, method="pca", n_components=None, min_ev=None, 
                      fa_method="minres", fa_rotation="promax",
                      seed=None):
    
    # to array
    data = np.array(data)
    
    # set n_components to max number if min explained variance is given
    n_components = data.shape[1] if (n_components is None) | (min_ev is not None) else n_components
    lgr.info(f"Performing dimensionality reduction using {method} (max components: "
             f"{n_components}, min EV: {min_ev}).")
    
    # case pca
    if method=="pca":
        # run pca with all components
        pcs = PCA(n_components=n_components).fit_transform(data)
        ev = np.var(pcs, axis=0) / np.sum(np.var(data, axis=0))
        # find number of components that sum up to total EV of >= min_ev
        if min_ev is not None:
            total_ev = 0
            for i, e in enumerate(ev):
                total_ev += e
                if total_ev>=min_ev:
                    n_components = i+1
                    lgr.info(f"{n_components} PC(s) explain(s) a total variance of "
                             f"{np.sum(ev[:n_components]):.04f} >= {min_ev} ({ev[:n_components]}).")
                    break
        # cut components & ev
        components = pcs[:,:n_components]
        ev = ev[:n_components]
        lgr.info(f"Returning {n_components} principal component(s).")
    
    # case ica
    elif method=="ica":
        components = FastICA(n_components=n_components, random_state=seed, max_iter=1000)\
            .fit_transform(data) 
        ev = None
        lgr.info(f"Returning {n_components} independent component(s).")
 
    # case fa
    elif method=="fa":
        # find number of components wihtout rotation that sum up to total EV of >= min_ev
        if min_ev is not None:
            fa = FactorAnalyzer(n_factors=n_components, method=fa_method, rotation=None)
            fa.fit(data)
            ev = fa.get_factor_variance()[2]
            if ev[-1]<min_ev:
                n_components -= 1
                lgr.warning(f"Given min EV ({min_ev}) > max possible EV ({ev[-1]:.02f})! "
                            f"Using max factor number ({n_components}).") 
            else:
                n_components = [i for i in range(len(ev)) if (ev[i] > min_ev)][1]
                lgr.info(f"{n_components} factor(s) explain(s) a total variance of "
                         f"{ev[n_components]:.02f} >= {min_ev}.")
        # run actual factor analysis
        fa = FactorAnalyzer(n_factors=n_components, method=fa_method, rotation=fa_rotation)
        fa.fit(data)
        components = fa.transform(data)
        loadings = fa.loadings_
        ev = fa.get_factor_variance()[1]
        lgr.info(f"Returning {n_components} factor(s).")
        
    else:
        lgr.critical(f"method = '{method}' not defined!")
    
    # get PCA and ICA "loadings"
    if method in ["pca", "ica"]:
        loadings = np.zeros((data.shape[1], n_components))
        for c in range(n_components):
            loadings[:,c] = corr(x=data,
                                 y=components[:,c:c+1], 
                                 rank=False, 
                                 correlate="c")[-1,:-1]
    
    ## return
    return components, ev, loadings


def null_to_p(test_value, null_array, tail="two", symmetric=False):
    """Return p-value for test value(s) against null array.
    
    Copied from NiMARE v0.0.12: https://zenodo.org/record/6600700
    (NiMARE/nimare/stats.py)
    
    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : bool
        When tail="two", indicates how to compute p-values. When False (default),
        both one-tailed p-values are computed, and the two-tailed p is double
        the minimum one-tailed p. When True, it is assumed that the null
        distribution is zero-centered and symmetric, and the two-tailed p-value
        is computed as P(abs(test_value) >= abs(null_array)).
    
    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).
    
    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.
    When the null distribution is known to be symmetric and centered on zero,
    and two-tailed p-values are desired, use symmetric=True, as it is
    approximately twice as efficient computationally, and has lower variance.
    """
    
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the unique
    # values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side="left").astype(float)
        return 1 - idx / len(null)

    if tail == "two":
        if symmetric:
            p = compute_p(np.abs(test_value), np.abs(null_array))
        else:
            p_l = compute_p(test_value, null_array)
            p_r = compute_p(test_value * -1, null_array * -1)
            p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(test_value * -1, null_array * -1)
    else:
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


def mc_correction(p_array, alpha=0.05, method="fdr_bh", how="array", dtype=None):
    
    # prepare data
    p = np.array(p_array)
    p_shape = p.shape

    ## correct across whole input array -> flattern & reshape
    if how in ["a", "arr", "array"]:
        # flatten row-wise
        p_1d = p.flatten("C") 
        # get corrected p-values
        res = multipletests(p_1d, alpha=alpha, method=method)
        pcor_1d = res[1]
        reject_1d = res[0]
        # reshape to original form
        pcor = np.reshape(pcor_1d, p_shape, "C")
        reject = np.reshape(reject_1d, p_shape, "C")
    
    ## correct across each column/row
    else:
        pcor, reject = np.zeros_like(p, dtype=dtype), np.zeros_like(p, dtype=dtype)
        if how in ["c", "col", "cols", "column", "columns"]:
            for col in range(p.shape[1]):
                res = multipletests(p[:,col], alpha=alpha, method=method)
                pcor[:,col], reject[:,col] = res[1], res[0]
        elif how in ["r", "row", "rows"]:
            for row in range(p.shape[0]):
                res = multipletests(p[row,:], alpha=alpha, method=method)
                pcor[row,:], reject[row,:] = res[1], res[0]
        else:
            print(f"Input how='{how}' not defined!")
          
    ## return as input dtype
    if isinstance(p_array, pd.DataFrame):
        pcor = pd.DataFrame(
            pcor, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
        reject = pd.DataFrame(
            reject, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
    elif isinstance(p_array, pd.Series):
        pcor = pd.Series(
            pcor, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)
        reject = pd.Series(
            reject, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)     
    return pcor, reject


def check_pearson(x,y):
    from scipy.stats import pearsonr
    r, _ = pearsonr(x,y)
    return r

def check_spearman(x,y):
    from scipy.stats import spearmanr
    r, _ = spearmanr(x,y)
    return r

def check_partialpearson(x,y,z):
    from pingouin import partial_corr
    data = pd.DataFrame(dict(x=x, y=y, z=z))
    res = partial_corr(data, "x", "y", "z", method="pearson")
    return res.r.values

def check_partialspearman(x,y, z):
    from pingouin import partial_corr
    data = pd.DataFrame(dict(x=x, y=y, z=z))
    res = partial_corr(data, "x", "y", "z", method="spearman")
    return res.r.values

def check_slr(x,y, adj_r2=True):
    import statsmodels.api as sm
    mlr = sm.OLS(y, sm.add_constant(x)).fit()
    if adj_r2:
        return mlr.rsquared_adj
    else:
        return mlr.rsquared
    
def check_beta(x,y):
    import statsmodels.api as sm
    mlr = sm.OLS(y, sm.add_constant(x), missing="drop").fit()
    return mlr.params[1:]

def check_dominance(x,y, return_stats="Total Dominance"):
    import contextlib

    from dominance_analysis import Dominance
    X = pd.DataFrame(x, columns=[f"pred{i}" for i in range(x.shape[1])])
    Y = pd.Series(y, name="target")
    df=pd.concat([X,Y], axis=1)
    with contextlib.redirect_stdout(None):
        dom = Dominance(data=df, 
                        target="target",
                        objective=1,
                        top_k=x.shape[1])
        dom.incremental_rsquare()
        dom_stats = dom.dominance_stats()
    return dom_stats[return_stats].sort_index().values

def check_residuals(x,y):
    import statsmodels.api as sm
    mlr = sm.OLS(y, sm.add_constant(x), missing="drop").fit()
    return mlr.resid
