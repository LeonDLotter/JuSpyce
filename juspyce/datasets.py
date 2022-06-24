import nilearn as nl

def get_template(template="T1", resolution=2, threshold=None):
    
    ## case MNI152 grey matter template or mask
    if template in ["gm", "GM", "gmv", "GMV"]:
        if threshold is None:
            temp = nl.datasets.load_mni152_gm_template(resolution=resolution)
        else:
            temp = nl.datasets.load_mni152_gm_mask(resolution=resolution, threshold=threshold)
            
    ## case MNI152 T1 template or brainmask
    elif template ["T1", "mask", "brainmask"]:
        if threshold is None:
            temp = nl.datasets.load_mni152_template(resolution=resolution)
        else:
            temp = nl.datasets.load_mni152_brain_mask(resolution=resolution, threshold=threshold)
            
    ## case not defined
    else:
        ValueError(template)
        
    ## return
    return temp

    
