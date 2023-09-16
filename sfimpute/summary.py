import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from .impute import to_dist_mat


def to_double_dist_mat(val1, val2):
    """upper right is val1, lower left is val2.

    Args:
        val1 (array): the first pairwise distance matrix.
        val2 (array): the second pairwise distance matrix.

    Returns:
        array: K*K matrix.
    """
    mat = to_dist_mat(val1)
    mat[np.tril_indices_from(mat)] = to_dist_mat(val2)[np.tril_indices_from(mat)]
    return mat


def heatmap_wrapper(
        val, ax, name1=None, name2=None, title=None, 
        cbar_kws={"shrink": 0.9}, **kwargs
):
    """wrapper of sns.heatmap with preferred parameters.

    Args:
        val (array): K*K matrix to plot.
        ax (plt.ax): plt.ax.
        name1 (str, optional): x label. Defaults to None.
        name2 (str, optional): y label. Defaults to None.
        title (str, optional): title. Defaults to None.
        cbar_kws (dict, optional): cbar_kws. Defaults to {"shrink": 0.9}.
    """
    tril = np.tril(val)
    if np.sum((~np.isnan(tril))&(tril != 0)) == 0:
        val = val.copy() # avoid changing the input values
        val[np.tril_indices_from(val)] = val.T[np.tril_indices_from(val)]

    cmap = plt.get_cmap("seismic_r")
    cmap.set_bad("white")
    sns.heatmap(
        val, cmap=cmap, square=True,
        xticklabels=False, yticklabels=False, #rasterized=True,
        ax=ax, cbar_kws=cbar_kws, **kwargs
    )
    ax.set_xlabel(name1, fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(name2, fontsize=10)
    ax.set_title(title, y=-0.2, fontsize=10)


def row_heatmap(mat_ls, cbar=False, **kwargs):
    """plot a row of heatmaps.

    Args:
        mat_ls (list): list of 1D or 2D arrays.
        cbar (bool, optional): whether to plot color bars. Defaults to False.
    """
    nmaps = len(mat_ls)
    if nmaps == 1:
        ax = plt.subplots(figsize=(3, 3))[1]
        axes = np.array([ax])
    else:
        axes = plt.subplots(nrows=1, ncols=nmaps, figsize=(3*nmaps, 3))[1]
    for i, ax in enumerate(axes.flat):
        hm = mat_ls[i] if len(mat_ls[i].shape) == 2 else to_dist_mat(mat_ls[i])
        heatmap_wrapper(hm, cbar=cbar, ax=ax, **kwargs)
