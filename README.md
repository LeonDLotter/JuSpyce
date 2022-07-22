# <a name="top"></a>JuSpyce - a toolbox for flexible assessment of spatial associations between brain images


[![DOI](https://zenodo.org/badge/473223442.svg)](https://zenodo.org/badge/latestdoi/473223442)  
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-sa/4.0/)  

---

*Spatial patterns observed in functional magnetic resonance images likely reflect function and dysfunction of underlying biological systems. This applies alike to function and structure, on the surface or in the volumetric space, to typical activation patterns, or to maps of disordered brain function, for example obtained from patients compared to a reference cohort.*

To date, there are several toolboxes targeting the question how brain images obtained from fMRI on the one hand and, e.g., nuclear imaging on the other can be related to another.  
The [JuSpace](https://github.com/juryxy/JuSpace) toolbox is especially suited for the correlation of nuclear imaging maps with subject-level *difference maps* obtained from fMRI by, e.g., using the mean activation pattern of one cohort (healthy controls) as a reference which is subtracted from each individual brain map in a target cohort (patients). Significance is then assessed by permutation of group labels.  
[BrainSMASH](https://brainsmash.readthedocs.io/en/latest/) is a toolbox that creates surrogate maps from parcellated or voxel-/vector-level imaging data which match the spatial autocorrelation pattern of the original maps. These null maps can then be used to generate empirical p values reflecting the probability of an alignment between, e.g., a PET map and a fMRI map being due to random spatial associations.  
The [neuromaps](https://netneurolab.github.io/neuromaps/) toolbox integrates several of these "surrogate map techniques" extended by advanced functions to transform imaging data between volumetric and surface spaces and statistical tests.

## JuSpyce

*JuSpyce* is a Python script version of the Matlab-based [JuSpace](https://github.com/juryxy/JuSpace) toolbox. The concept is based on the toolboxes mentioned above with the following line of thought. 

### 1. The data: `JuSpyce.fit()`

We have volumetric or surface neuroimaging data in original space or already parcellated. If not parcellated, we provide a volumetric or surface atlas to parcellate these original images.
Now, we build two (three) dataframes in which parcels are stored in the columns and "maps" (PET maps, subject maps, ...) are stored in the rows.
- **`X`**: A dataframe that contains data treated as "predictors" (independent variables). This would be, for example, nuclear imaging data or data generated from the Allen Brain Atlas, etc. 
- **`Y`**: A dataframe that contains data treated as "targets" (dependent variables). This would be single subject data, or an average brain map obtained from a certain fMRI task, or the mean gray matter volume in a reference cohort, or a meta-analytic image, and so on.
- (**`Z`**: An optional dataframe containing data that should be "regressed out" of the data above when correlating, for example, `X` and `Y` data using partial correlations.)

#### 1.1 Data transforms: `JuSpyce.transform()`

The data (`X` or `Y`) can be "transformed" in the following ways:
- **Parcelwise mean**: The parcelwise mean of the whole dataset (`mean`).
- **Confound regression**: Data in `Z` is regressed out of either `X` or `Y` and the residuals from this regression replace `X` or `Y`(`partial`).
- **Dimensionality reduction**: We can run PCA, ICA, or factor analysis with fixed numbers of factors or retained-explained-variance thresholds (for PCA and factor analysis) to reduce the number of predictors/targets. This is especially relevant for the predictors (`X`), if these show string intercorrelation patterns (`pca`, `ica`, `fa`).

#### 1.2 Data comparison: `JuSpyce.compare()`

The `Y` data can be "compared" between predefined groups in the following ways. The idea to this is based on the JuSpace core functionality. We have to supply a group assignment vector (python list of nulls and ones).
- **One group minus the mean of the other**: The difference between each vector of group A and the parcelwise means of group B (`diff(A,mean(B))`, `diff(B,mean(A))`).
- **Mean difference**: The difference between the parcelwise means of group A and the parcelwise means of group B (`diff(mean(A),mean(B))`).
- **Effect sizes of the group difference**: The parcelwise effect sizes (Cohen's d, Hedge's g, paired Cohen's d) of group A compared with group B (`cohen(A,B)`, `hedge(A,B)`, `pairedcohen(A,B)`).

### 2. "Prediction": `JuSpyce.predict()`

Let's assume, we are interested in how our `X` and `Y` data relate directly to each other. We can answer our questions with several options. All the methods mentioned below are implemented using custom code based on vectorized operations that are faster for our purpose then the usual python implementations.
- **Correlation**: We correlate each `X` with each `Y` using (partial) Spearman or Pearson correlations (`spearman`, `pearson`, `partialspearman`, `partialpearson`).
- **Univariate regression**: We "predict" each `Y` from each `X` and retain the individually explained variance $R^2$ (`slr`).
- **Multivariate regression**: We "predict" each `Y` from all `X` at once. We retain the overall explained variance, the predictorwise (`X`) beta coefficients and an approximation of the predictorwise contribution to the overall explained variance calculated as the difference between the total $R^2$ and the $R^2$ as obtained from all `X` without the predictor in question (`mlr`).
- **Dominance analysis**: As in the multivariate regression but with the very nice feature that the exact individual contribution of each `X` to the overall explained variance is quantified. This is done by calculating all possible combinations of predictors and calculating summary metrics ("dominance statistics"). The computation time for this approach scales exponentially with the number of predictors. Going far beyong 15 predictors will be difficult here.

### 3. Significance 

#### 3.1 Based on spatial null maps: `JuSpyce.permute_maps()`

To assign nonparametric and spatial autocorrelation-corrected p values to each "prediction metric" from the point above, we can generate surrogate ("null") maps and rerun `JuSpyce.predict()` on these to obtain null distributions corresponding to each "true" prediction metric. From these null distributions, empirical p values are then calculated. Null maps can be created for either `X` or `Y` data, or both at once. The typical approach would be to use the predictor data `X`. Datawise, we can use:
- The "raw" `X` and `Y` input data
- The "new" `X` data after, e.g., dimensionality reduction
- The "new" `Y` data after, e.g., the parcelwise effect size between two defined groups has been calculated. This analysis would show us if the difference map between two groups can be "predicted" from our, e.g. PET data in comparison to predictor maps with similar spatial properties.

#### 3.2 Based on group permutation: `JuSpyce.permute_groups()`

In accordance with the original JuSpace approach, we can also test whether the difference between two groups in the `Y` data is significantly associated to predictors in the `X` data by permuting the group labels and rerunning `JuSpyce.compare()` and `JuSpyce.predict()`. 
- Running this on, e.g., the parcelwise effect size between groups would answer a question related to the last point in 3.1, i.e., whether the spatial relation between the difference map of two groups and the predictors is stronger than would be assumed if the group allocation had no meaning.
- Running this using the individual vectors of one group in relation to the parcelwise means of another (treated as "reference") would answer a comparable question but the subject-level values could be used for further follow-up analyses.

#### 3.3 Multiple comparison correction: `JuSpyce.correct_p()`

Empirical p values can be corrected by running `JuSpyce.correct_p()` either with `analysis=predictions` to correct the values obtained from `JuSpyce.permute_maps()` or with `analysis=comparisons` to correct p values calculated with `JuSpyce.permute_groups()`.

## Practical usage

JuSpyce is in the development stage. There will be bugs - fell free to open an issue!  
There is currently no documentation integrated in the code and non available elsewhere. Jupyter notebooks using the functionality refered to above, along with example data obtained from neuromaps and [Neuroquery](https://neuroquery.org/), are available in the [testing](/testing/) folder.  
A thought out example case, detailed documentation with API references, pip-integration, and a paper will follow in time. I also plan to add integrated datasets, vizualisation functions, and I am aware that the code is a bit messy at the moment...

### Simple example:

```python
# import JuSpyce
import sys
sys.path.append("path/to/juspyce/folder")
from juspyce.api import JuSpyce

## initialize
juspyce_object = JuSpyce(
  x=predictor_list, # list of volumetric data, or df with shape(n_data, n_parcels)
  y=target_list, # list of volemtric data or df with shape(n_data, n_parcels
  data_space="MNI152", # "MNI152", "fsaverage" or "fslr"
  parcellation=parcellation_volume, # used parcellation
  parcellation_labels=parcellation_volume_labels, # parcel labels as list
  parcellation_space="MNI152", # space of parcellation
  standardize=True, # will z-transform all data. Can also be "x", "xy", "xyz"
  n_proc=8 # number of parallel processes for various methods
)
juspyce_object.fit()

## predict 
# spearman correlations
juspyce_object.predict( 
  method="spearman", 
  r_to_z=True, # apply r to Z transformation
  n_proc=1 # 1 process will be faster here
)
print(juspyce_object.predictions["spearman"])

## permute maps
juspyce_object.permute_maps(
  method="spearman", # which method
  permute="X", # null maps for which data? Can be 'X' or 'Y'
  null_method="variogram", # "variogram" -> brainsmash, "random" -> np.random
  n_perm=1000, # number of permutations (= number of null maps)
  r_to_z=True,
  n_proc=8, # number of processes
  seed=41, # seed for reproducability
)
print(juspyce_object.p_predictions["spearman"])

## compare
juspyce_object.compare(
  comparison="diff(A,mean(B))", 
  groups=group_list # list with zeros and ones assigning targets to two groups
)
print(juspyce_object.comparisons["diff(A,mean(B))"])

## permute groups
juspyce_object.permute_groups(
  groups=group_list,
  comparison="diff(A,mean(B))" # difference between individual A and B
  method="spearman", # which method
  n_perm=1000, # number of permutations (= number of null maps)
  r_to_z=True,
  n_proc=8, n_proc_predict=1, seed=41
)
print(juspyce_object.p_comparisons["diff(A,mean(B))-spearman"])

## correct p values
juspyce_object.correct_p(
  analysis="comparisons", # or "predictions"
  mc_method="fdr_bh" # correction method passed to statsmodels
)
print(juspyce_object.p_comparisons["diff(A,mean(B))-spearman--fdr_bh"])
```
### Testing notebooks (with examples)

- [juspyce.fit()](/testing/test_1_juspyce.fit.ipynb)
- [juspyce.transform()](/testing/test_2_juspyce.transform.ipynb)
- [juspyce.compare()](/testing//test_3_juspyce.compare.ipynb)
- [juspyce.predict()](/testing/test_4_juspyce.predict.ipynb)
- [juspyce.permute_maps()](/testing/test_5_juspyce.permute_maps.ipynb)
- [juspyce.permute_groups()](/testing/test_6_juspyce.permute_groups.ipynb)

## Why this name?

1. The origin of both JuSpace and JuSpyce is the Jülich Research Centre, Germany. Literally any stuff coming from there starts with "Ju". 
2. JuSpace got its "Space" from focus on spatial correlations. 
3. Things in Python have to have a "py" in the name. 
4. People (that's me) have said that JuSpyce is a spyced up version of JuSpace.
5. Voilà: *JuSpyce*

## Contact

Do you have questions, comments or suggestions, or would like to contribute to the toolbox? [Contact me](mailto:leondlotter@gmail.com)! 

---
[Back to the top](#top)


