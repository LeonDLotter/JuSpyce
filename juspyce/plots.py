import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def hide_empty_axes(axes):
    
    [ax.axis("off") for ax in axes.ravel() if ax.axis() == (0.0, 1.0, 0.0, 1.0)]


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)
