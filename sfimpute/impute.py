from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats, interpolate, signal
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from skimage import transform
from sklearn.metrics.pairwise import nan_euclidean_distances

class DistDataFrame(pd.DataFrame):
    """extends pd.DataFrame to store pairwise distances."""
    def __init__(self, *args, **kwargs):
        """initialize a DistDataFrame. Convert all column names to str."""
        super().__init__(*args, **kwargs)
        # convert all cell_ids to string
        self.columns = self.columns.astype("str")

    @property
    def _constructor(self):
        """ensures all following operations return DistDataFrame.

        Returns:
            type: DistDataFrame.
        """
        return DistDataFrame

    def kcol(self):
        """returns the names of the information columns.

        Returns:
            np.array: key column names.
        """
        COL = ["chr", "bin1", "x1", "bin2", "y1", "y-x", "lmbda", "mu", "var"]
        return np.array([t for t in self.columns if t in COL])

    def vcol(self):
        """returns the names of the chromosomes.

        Returns:
            np.array: value column names.
        """
        COL = ["chr", "bin1", "x1", "bin2", "y1", "y-x", "lmbda", "mu", "var"]
        return np.array([t for t in self.columns if t not in COL])

    def to_darr(self, vcol=None):
        """return the value columns as numpy array.

        Args:
            vcol (list, optional): cell_ids. Defaults to None.

        Returns:
            np.ndarray: the pairwise distances.
        """
        if vcol is None:
            return self[self.vcol()].values
        else:
            return self[vcol].values


def to_dist_mat(d, fill_val=0.0):
    """convert a 1-D distance vector to matrix.

    Args:
        d (np.array): 1-D pairwise distances.
        fill_val (float): values to fill the diagonal. Defaults to 0.0.

    Returns:
        arr: K*K dimension matrix.
    """
    num_ids = int((1 + (1+8*len(d))**0.5)/2)
    id_iter = combinations(np.arange(1, num_ids + 1), 2)
    mat_df = pd.DataFrame(list(id_iter), columns=["id1", "id2"])
    mat_df["d"] = d
    dist_vals = pd.pivot_table(mat_df, "d", "id1", "id2", dropna=False).values
    dist_mat = np.diag([fill_val]*num_ids)
    dist_mat[np.triu_indices(num_ids, 1)] = dist_vals[np.triu_indices_from(dist_vals)]
    dist_mat[np.tril_indices(num_ids, -1)] = dist_vals.T[np.tril_indices_from(dist_vals)]
    return dist_mat


def to_dist_df(coor_data, ann_df):
    """convert 3D coordinates to pairwise distances.

    Args:
        coor_data (pd.DataFrame): coordinate df with columns chr, cell_id,
        pos, x, y, and z, have NaNs.
        ann_df (pd.DataFrame): annotation file.

    Returns:
        pd.DataFrame: (chr*M)*(5+N).
    """
    coor_dists = coor_data.groupby(
        ["chr", "cell_id"], sort=False
    ).apply(
        lambda x: pdist(x[["x", "y", "z"]])
    )

    dist_df_ls = []
    for c, sub_ann_df in ann_df.groupby("chr", sort=False):
        single_chr_dists = np.stack(coor_dists[c].values).T
        sub_dist_df = pd.DataFrame(single_chr_dists, columns=coor_dists[c].index)

        ann_comb = combinations(sub_ann_df[["pos", "start"]].values, 2)
        bin_cols = np.stack(list(ann_comb)).reshape((-1,4))
        sub_bin_df = pd.DataFrame(bin_cols, columns=["bin1", "x1", "bin2", "y1"])
        sub_bin_df.insert(0, "chr", c)

        dist_df_ls.append(pd.concat([sub_bin_df, sub_dist_df], axis=1, sort=False))

    dist_df = pd.concat(dist_df_ls, axis=0, sort=False)
    cell_ids = pd.unique(coor_data["cell_id"]).tolist()
    sorted_cols = dist_df.columns[:sub_bin_df.shape[1]].tolist() + cell_ids
    dist_df = DistDataFrame(dist_df.loc[:,sorted_cols])
    dist_df.insert(
        len(dist_df.kcol()), 
        "y-x",
        dist_df["y1"]-dist_df["x1"]
    )
    start1d = ann_df["start"]
    median = np.nanmedian(start1d - start1d.shift())
    median = round(median, 2 - len(str(int(median))))
    dist_df["y-x"] = (dist_df["y-x"]//median * median).astype("int")
    return dist_df


def to_triu_indices(n, k):
    """find the upper triangle indices of an array of square matrices.

    Args:
        n (int): length in the first dimension.
        k (int): length in the second and the third dimension.

    Returns:
        tuple: (idx1, idx2, idx3)
    """
    chr_idxs = np.arange(n, dtype="int")
    tri_idxs1, tri_idxs2 = np.triu_indices(k, 1)
    return (
        np.repeat(chr_idxs, tri_idxs1.shape[0]),
        np.tile(tri_idxs1, chr_idxs.shape[0]),
        np.tile(tri_idxs2, chr_idxs.shape[0])
    )


def to_mats_single_reg(vec_arr):
    """convert flattened 1D distance vectors to 2D matrices. The reverse operation of 
    to_vecs_single_reg.

    Args:
        vec_arr (np.ndarray): of shape (n, C^k_2), pairwise distances/dissimilarities of a single 
        imaging region.

    Returns:
        np.ndarray: of shape (n, k, k)
    """
    n = vec_arr.shape[0]
    k = int((1 + (1+8*vec_arr.shape[1])**0.5)/2)
    # diagonal and lower left triangle are filled by np.nan
    mat_arr = np.ones((n, k, k))*np.nan
    idxs = to_triu_indices(n, k)
    mat_arr[idxs] = vec_arr.ravel()
    return mat_arr


def to_vecs_single_reg(mat_arr):
    """roll out the upper triangle of pairwise distances. The reverse operation of 
    to_mats_single_reg.

    Args:
        mat_arr (np.ndarray): of shape (n, k, k)

    Returns:
        np.ndarray: of shape (n, C^k_2)
    """
    n, k = mat_arr.shape[:2]
    idxs = to_triu_indices(n, k)
    return mat_arr[idxs].reshape((n, -1))


def conv_resize(mat, rsize):
    """smooth and resize a square matrix.

    Args:
        mat (np.ndarray): k*k matrix.
        rsize (int, optional): resized factor.

    Returns:
        np.ndarray: smoothed and resized matrix.
    """
    nan_pos = np.isnan(mat)
    v = np.array([np.where(nan_pos, 0, mat), np.where(nan_pos, 1, 0)])
    rlv = transform.downscale_local_mean(v, (1, rsize, rsize))
    rlv[0] = np.where(rlv[1]!=1, rlv[0], np.nan)
    r = rlv[0]/(1 - rlv[1])
    np.fill_diagonal(r, np.nan)
    return r


def read_data(path):
    """read 3D coordinates/pairwise distances files. Convert imaging region and cell ID columns to 
    strings.

    Args:
        path (str): file path.

    Returns:
        pd.DataFrame: data read from the file path.
    """
    dtype_dict = {"chr":"str", "cell_id":"str", "pos":"int"}
    data = pd.read_csv(path, sep="\t")
    shared_type = {k:v for k,v in dtype_dict.items() if k in data.columns}
    data = data.astype(shared_type)
    return data


def save_coor(save_coor_df, raw_coor_path, save_path):
    """save imputed 3D coordinates. Automatically round the 3D coordinates to the precision of the 
    input file.

    Args:
        save_coor_df (pd.DataFrame): the dataframe to save.
        raw_coor_path (pd.DataFrame): the raw 3D coordinates path.
        save_path (str): path to save the file.
    """
    raw_as_str = pd.read_csv(raw_coor_path, dtype="str", sep="\t")
    for coor_name in ["x", "y", "z"]:
        str_col = raw_as_str[coor_name].astype("str")
        precs = str_col.str.replace(r"^[^\.]+\.?", "").str.len()
        prec = np.max(precs)
        save_coor_df[coor_name] = save_coor_df[coor_name].round(prec)
    save_coor_df.to_csv(save_path, sep="\t", index=False)


def insert_missing_rows_single_chr(data, ann):
    """helper method of insert_missing_rows, insert all missing rows of a single imaging region.

    Args:
        data (pd.DataFrame): coordinate df with columns chr, cell_id, pos, x, y, and z of the 
        imaging region.
        ann (pd.DataFrame): annotation file of the imaging region.
    """
    CELL_IDS = pd.unique(data["cell_id"])
    NUM_POS = ann.shape[0]

    cell_ids = np.repeat(CELL_IDS, NUM_POS)
    pos_ids = np.tile(ann["pos"], len(CELL_IDS))
    full_rids = set(zip(cell_ids, pos_ids))
    raw_rids = set(zip(data["cell_id"], data["pos"]))
    # expected (cell id, locus id) - observed (cell id, locus id)
    missed_rows = full_rids - raw_rids
    
    if len(missed_rows) != 0:
        missed_rows = np.array(list(missed_rows))
        missed_df = pd.DataFrame(
            missed_rows, 
            columns=["cell_id", "pos"]
        )
        missed_df["pos"] = missed_df["pos"].astype("int")
        data = pd.concat([data, missed_df], sort=False)
    
    # insert all missing cell id and locus id
    data = data.sort_values(["cell_id", "pos"])
    return data.reset_index(drop=True)


def insert_missing_rows(coor_df, ann_df):
    """fill all missing loci by NaN.

    Args:
        coor_df (pd.DataFrame): coordinate df with columns chr, cell_id, pos, x, y, and z of all 
        imaging regions.
        ann_df (pd.DataFrame): annotation file of all imaging regions.
    """
    inserted_ls = []
    for im_id, data in coor_df.groupby("chr", sort=False):
        ann = ann_df[ann_df["chr"]==im_id]
        sub_df = insert_missing_rows_single_chr(data, ann)
        sub_df["chr"] = im_id
        inserted_ls.append(sub_df)
    return pd.concat(inserted_ls, sort=False)


def interpolate_coors_scipy(ann_df, coor_df, kind):
    """impute missing 3D coordinates by scipy's interp1d. The independent variable is the starting 
    1D genomic location of each locus.

    Args:
        ann_df (pd.DataFrame): 1D genomic location annotation file.
        coor_df (pd.DataFrame): 3D coodinates data with missing values as NaN.
        Contains chr, cell_id, pos, x, y, and z as columns.
        kind (str): interpolation method, argument passed to scipy's interp1d.

    Returns:
        pd.DataFrame: same shape as coor_df with all missing values filled by scipy's interp1d.
    """
    interp_vals = []
    for (im_id, cell_id), df in coor_df.groupby(["chr", "cell_id"], sort=False):
        x = ann_df[ann_df["chr"]==im_id]["start"].values
        y = df[["x", "y", "z"]].values
        nan_filter = ~np.isnan(y[:, 0])
        interp_fxn = interpolate.interp1d(
            x[nan_filter], y[nan_filter, :], axis=0,
            kind=kind, fill_value="extrapolate"
        )
        interp_vals.append(interp_fxn(x))
    interp_vals = np.concatenate(interp_vals)
    interp_coor_df = coor_df.copy()
    interp_coor_df[["x", "y", "z"]] = interp_vals
    return interp_coor_df


def generate_interpolation(ann_path, coor_path, kind, save_path):
    """generate the interpolated 3D coordinates file.

    Args:
        ann_path (str): path to the 1D genomic location annotation file.
        coor_path (str): path to the 3D coordinates file. Contains chr, cell_id, pos, x, y, and z as
        columns. Missing values are filled by NaN.
        kind (str): interpolation method, argument passed to scipy's interp1d.
        save_path (str): the path to save the imputation result.
    """
    ann_df = read_data(ann_path)
    coor_df = read_data(coor_path)

    interp_df = interpolate_coors_scipy(ann_df, coor_df, kind)
    interp_df.to_csv(save_path, sep="\t", index=False)


def boxcox_by1d(by1d_arr):
    """transform to normal by box-cox transformation and then to N(0,1).

    Args:
        by1d_arr (np.ndarray): rows of normalized pairwise distances.

    Returns:
        np.ndarray, np.ndarray: transformed data, transformed parameters.
    """
    y = by1d_arr[~np.isnan(by1d_arr)]
    lmbda = round(stats.boxcox_normmax(y, method="mle"), 4)
    y_trf = stats.boxcox(y, lmbda=lmbda)
    mu, var = np.mean(y_trf), np.var(y_trf)
    by1d_arr[~np.isnan(by1d_arr)] = (y_trf - mu)/(var**0.5)
    trf_vals = np.tile([lmbda, mu, var], by1d_arr.shape[0]).reshape(-1, 3)
    return by1d_arr, trf_vals


def normalize_pdist_by1d(dist_df):
    """normalize all pairwise distances by 
    For each chr:
    1) box-cox transformation for each set of bin pairs with the same 1D genomic distances
    2) transform all distributions to N(0,1)

    Args:
        dist_df (pd.DataFrame): dist_df of a all chromosomes.

    Returns:
        pd.DataFrame: dist_df-like object with normalized distances.
    """
    dist_df_ls = []
    for c, sub_dist_df in dist_df.groupby("chr", sort=False):

        kcol_ls, norm_pdists, trf_val_ls = [], [], []
        for d, df in sub_dist_df.groupby("y-x", sort=False):
            by1d_arr, trf_vals = boxcox_by1d(df.to_darr())
            kcol_ls.append(df[df.kcol()].reset_index(drop=True))
            norm_pdists.append(by1d_arr)
            trf_val_ls.append(trf_vals)

        kcols = pd.concat(kcol_ls, sort=False, ignore_index=True)
        norm_pdists = pd.DataFrame(np.concatenate(norm_pdists), columns=sub_dist_df.vcol())
        trf_vals = pd.DataFrame(np.concatenate(trf_val_ls), columns=["lmbda", "mu", "var"])
        n_df = DistDataFrame(pd.concat([kcols, trf_vals, norm_pdists], sort=False, axis=1))
        n_df = n_df.sort_values(["bin1", "bin2"])
        dist_df_ls.append(n_df)
        
    return pd.concat(dist_df_ls, ignore_index=True, sort=False)


def inverse_trf(chr_norm_df):
    """inverse transformation, N(0,1) -> N(m, s) -> inv_boxcox.

    Args:
        chr_norm_df (pd.DataFrame): normalized pairwise distances,
            column names: chr, bin1, x1, bin2, y1, lmbda, mu, var
            row names: combination of bin pairs

    Returns:
        pd.DataFrame: same format as chr_norm_df.
    """
    target_mat = chr_norm_df.to_darr()
    l, m, v = np.expand_dims(chr_norm_df[["lmbda", "mu", "var"]].values.T, 2)

    lne0 = np.exp(np.log((target_mat*np.sqrt(v) + m)*l + 1)/l)
    le0 = np.exp(target_mat*np.sqrt(v) + m)
    l_filter = np.repeat(l, target_mat.shape[1], axis=1) == 0

    target_mat_trf = np.where(l_filter, le0, lne0)
    target_df = chr_norm_df.copy()
    target_df[target_df.vcol()] = target_mat_trf
    return target_df


def to_simil_mat_row(pdist_arr, i):
    """calculate the ith row of the dissimilarity matrix.

    Args:
        pdist_arr (np.ndarray): N*M normalized pairwise distances.
        i (int): which row to calculate.

    Returns:
        np.ndarray: 2*N, dissimil_raw, count_ratio
    """
    num_chrs = pdist_arr.shape[0]
    tiled_dist = np.tile(pdist_arr[i,None].T, num_chrs).T
    diff = tiled_dist - pdist_arr
    eucl_dist = np.sqrt(np.nansum(np.square(diff), axis=1))
    share_count = np.sum(~np.isnan(diff), axis=1)
    dissimil_raw = eucl_dist/np.sqrt(share_count)

    by_chr_count = np.sum(~np.isnan(pdist_arr), axis=1)
    min1 = by_chr_count[i]
    min_count = np.where(min1>by_chr_count, by_chr_count, min1)
    count_ratio = share_count/min_count
    return np.stack([dissimil_raw, count_ratio])
    

def filter_by_prop(count_ratio, simil_mat, ratio):
    """filter pairwise dissimilarities by NaN proportions.

    Args:
        count_ratio (np.ndarray): N*N NaN ratios
        simil_mat (np.ndarray): N*N pairwise dissimilarities.
        ratio (float): between 0 and 1. Defaults to 0.8.

    Returns:
        np.ndarray: N*N filtered dissimilarities.
    """
    min_r = min(np.nanquantile(count_ratio, 0.99), ratio)
    simil_mat[count_ratio < min_r] = np.nan
    np.fill_diagonal(simil_mat, np.nan)
    return simil_mat


def to_simil_mat(pdist_arr, ratio=0.8):
    """calculate the pairwise similarities from pairwise distances with NaN entries. For any two 
    chromosomes, the similarity between them is only defined if the number of shared available 
    entries is larger than ratio% of the number of available entries in at least one chromosome.

    Args:
        pdist_arr (np.ndarray): N*M normalized pairwise distances.
        ratio (float): between 0 and 1. Defaults to 0.8.

    Returns:
        np.ndarray: N*N pairiwse similarities.
    """
    count_iter = combinations(np.sum(~np.isnan(pdist_arr), axis=1), 2)
    min_count = np.min(np.array(list(count_iter)), axis=1)
    share_count = pdist(~np.isnan(pdist_arr), lambda u,v: np.sum(u&v))
    count_ratio = to_dist_mat(share_count/min_count)

    simil_mat = nan_euclidean_distances(pdist_arr)
    simil_mat = simil_mat/(pdist_arr.shape[1]**0.5)

    return filter_by_prop(count_ratio, simil_mat, ratio)


def scatter_row_idxs(comm, nrows, MPI):
    """scatter row idxs to each process.

    Args:
        comm (MPI.COMM_WORLD): MPI COMM_WORLD.
        nrows (int): number of rows to scatter.
        MPI (MPI): MPI object.
    """
    if comm.rank == 0:
        row_idxs = np.arange(nrows, dtype="int64")
        ave, res = divmod(nrows, comm.size)
        count = np.array([ave+1 if p < res else ave for p in range(comm.size)])
        index = np.array([sum(count[:p]) for p in range(comm.size)])
    else:
        count = np.zeros(comm.size, dtype="int")
        index = np.zeros(comm.size, dtype="int")
        row_idxs = None
    comm.Bcast(count, root=0)
    comm.Bcast(index, root=0)
    rows_assigned = np.zeros(count[comm.rank], dtype="int64")
    comm.Scatterv([
        row_idxs, count, index, MPI.LONG
    ], rows_assigned, root=0)
    return rows_assigned, count, index


def simil_mat_parallel(pdist_arr, MPI):
    """scatter each row and return the dissimilarities to all processes.

    Args:
        pdist_arr (np.ndarray):  N*M normalized pairwise distances.
        MPI (MPI): MPI object.

    Returns:
        np.ndarray: N*N filtered pairwise dissimilarities.
    """
    comm = MPI.COMM_WORLD
    rows_assigned, count, index = scatter_row_idxs(comm, pdist_arr.shape[0], MPI)

    rows = np.stack([to_simil_mat_row(pdist_arr, i) for i in rows_assigned])
    simil_rows, count_rows = rows[:,0,:].ravel("C"), rows[:,1,:].ravel("C")

    nchrs = pdist_arr.shape[0]
    simil_all = np.zeros(nchrs**2, dtype="float64")
    comm.Gatherv(simil_rows, [simil_all, count*nchrs, index*nchrs, MPI.DOUBLE], root=0)
    simil_all = simil_all.reshape(nchrs, nchrs)

    count_all = np.zeros(nchrs**2, dtype="float64")
    comm.Gatherv(count_rows, [count_all, count*nchrs, index*nchrs, MPI.DOUBLE], root=0)
    count_all = count_all.reshape(nchrs, nchrs)

    if comm.rank == 0:
        dissimil_mat = filter_by_prop(count_all, simil_all, 0.8)
    else:
        dissimil_mat = np.zeros((nchrs, nchrs), dtype="float64")
    comm.Bcast(dissimil_mat, root=0)
    return dissimil_mat


def to_target_single_chr(pdist_arr, wr):
    """calculate the target distance matrix for a given chromosome based on a nn nearest entries.

    Args:
        pdist_arr (np.ndarray): N*M matrix of normalized pairwise distances.
        wr (np.array): 1*N, weight computed from similarities.

    Returns:
        np.array: 1*M, the predicted pairwise distances for the chromosome.
    """
    expanded_weight = np.repeat(wr.reshape((1, -1)), pdist_arr.shape[1], axis=0)
    expanded_weight[np.isnan(pdist_arr.T)] = 0

    thresh = -np.sort(-expanded_weight, axis=1)[:, 1]
    expanded_weight[expanded_weight <= thresh[:, None]] = 0
    weight = expanded_weight/expanded_weight.sum(axis=1)[:, None]

    pdists_wo_nan = np.where(np.isnan(pdist_arr), 0, pdist_arr)
    target = (weight*pdists_wo_nan.T).sum(-1)

    # entries where all cells with positive weights are NaN
    target[target==0] = np.nan
    return target


def to_target_parallel(pdist_arr, dissimil_mat, MPI):
    """fill target pairwise distances parallely.

    Args:
        pdist_arr (np.ndarray): N*M normalized pairwise distances.
        dissimil_mat (np.ndarray): N*N pairwise dissimilarities.
        MPI (MPI): MPI object.

    Returns:
        np.ndarray: N*M filled normalized pairwise distances.
    """
    comm = MPI.COMM_WORLD
    W = np.where(np.isnan(dissimil_mat), 0, 1/dissimil_mat)
    rows_assigned, count, index = scatter_row_idxs(comm, W.shape[0], MPI)
    pdists = np.array([
        to_target_single_chr(pdist_arr, W[i])
        for i in rows_assigned
    ]).ravel("C")

    target_pdists = np.zeros(pdist_arr.size, dtype="float64")
    n = pdist_arr.shape[1]
    comm.Gatherv(pdists, [target_pdists, count*n, index*n, MPI.DOUBLE], root=0)
    target_pdists = target_pdists.reshape(pdist_arr.shape)
    comm.Bcast(target_pdists, root=0)
    return target_pdists


def impute_pdist_one_iter(pdist_arr):
    """construct the predicted pairwise distance matrice.

    Args:
        pdist_arr (np.ndarray): N*M matrix, where
            N: number of chromosomes
            M: number of combination of bins

    Returns:
        np.ndarray, int: N*M predicted values and kernel size.
    """
    simil_mat = to_simil_mat(pdist_arr, ratio=0.8)
    W = np.where(np.isnan(simil_mat), 0, 1/simil_mat)
    target_pdists = np.array([to_target_single_chr(pdist_arr, w) for w in W])
    return target_pdists


def count_na_in_pdist(pdist_arr):
    """count the total number of NaN entries in pairwise distance matrix. Does not count chromosomes
    with all NaN values.

    Args:
        pdist_arr (np.ndarray): N*M pairwise distances.

    Returns:
        int: number of NaN entries.
    """
    avail_idx = np.sum(~np.isnan(pdist_arr), axis=1) != 0
    return np.sum(np.isnan(pdist_arr[avail_idx]))


def generate_single_reg_target(reg_norm_df, save_path):
    """generate the target matrices for a single imaging region.

    Args:
        reg_norm_df (DistDataFrame): normalized pairwise distances of a region,
            column names: chr, bin1, x1, bin2, y1, lmbda, mu, var, *chr_ids
            row names: combination of bin pairs
        save_path (str): the path of the generated file.
    """
    reg_id = reg_norm_df["chr"].iloc[0]
    reg_pred_df = reg_norm_df.copy()
    pdist_arr = reg_pred_df.to_darr().T

    np.seterr(divide="ignore", invalid="ignore")
    prev_na, curr_na = None, count_na_in_pdist(pdist_arr)
    print(f"Region {reg_id} initial: {curr_na} NaN values")
    while prev_na != curr_na and curr_na != 0:
        P = impute_pdist_one_iter(pdist_arr)
        pdist_arr = np.where(np.isnan(pdist_arr), P, pdist_arr)
        prev_na = curr_na
        curr_na = count_na_in_pdist(pdist_arr)
        print(f"Region {reg_id}: {curr_na} NaN values")
    np.seterr(divide="warn", invalid="warn")

    reg_pred_df[reg_pred_df.vcol()] = pdist_arr.T
    reg_pred_df.round(6).to_csv(save_path, sep="\t", index=False)


def loss_pdist(x, raw_coors, mats):
    """computes the loss by summing the squared difference of pairwise distances. 

    Args:
        x (np.array): flattend updated 3D coordinates.
        raw_coors (np.ndarray): K*3, raw 3D coordinates with NaN.
        mats (np.ndarray): 4*K*K, [target, lambda, mu, std].

    Returns:
        float: computed loss.
    """
    upt_coors = raw_coors.copy()
    upt_coors[np.isnan(upt_coors)] = x
    upt_pdist = to_dist_mat(pdist(upt_coors), np.nan)

    # transformed the distances by box-cox and normalization
    trf_upt_pdist = (np.power(upt_pdist, mats[1]) - 1)/mats[1]
    trf_upt_pdist = (trf_upt_pdist - mats[2])/mats[3]

    # return the sum of squares
    return np.nansum(np.square(trf_upt_pdist - mats[0]))


def jac_pdist(x, raw_coors, mats):
    """computes the Jacobian of the loss function.

    Args:
        x (np.array): flattend updated 3D coordinates.
        raw_coors (np.ndarray): K*3, raw 3D coordinates with NaN.
        mats (np.ndarray): 4*K*K, [target, lambda, mu, std].

    Returns:
        np.array: computed Jacobian, same shape as x.
    """
    upt_coors = raw_coors.copy()
    upt_coors[np.isnan(upt_coors)] = x
    upt_pdist = to_dist_mat(pdist(upt_coors), np.nan)

    # transformed the distances by box-cox and normalization
    trf_upt_pdist = (np.power(upt_pdist, mats[1]) - 1)/mats[1]
    trf_upt_pdist = (trf_upt_pdist - mats[2])/mats[3]

    # the second part of the Jacobian
    prod_part = np.power(upt_pdist, mats[1]-2)*(trf_upt_pdist - mats[0])
    prod_part = prod_part*2/mats[3]

    num_bins = upt_coors.shape[0]
    # expand the 3D coordinates to 3*K*K to compute pairwise differences
    upt_expand = np.repeat(upt_coors.T, num_bins).reshape((3, num_bins, -1))
    coor_diff = (upt_expand.transpose((0, 2, 1)) - upt_coors.T[:,:,None])

    jac = np.nansum(coor_diff*prod_part[None,:,:], axis=1).T
    return jac[np.isnan(raw_coors)]


def recover_3d_coor_single_chr(raw, lnr, fla_mat, trf):
    """recover the 3D coordinates based on a target pairwise distance matrix, use the transformed 
    loss.

    Args:
        raw (array): K*3, with NaN values.
        lnr (array): K*3, linear imputed values.
        fla_mat (array): C^n_2, target pairwise distances.
        trf (array): (C^n_2)*3, lmbda, mu, var columns.

    Returns:
        array: same shape as raw, recovered 3D coordinates.
    """
    upt = raw.copy()
    mats = np.stack([to_dist_mat(m, np.nan) for m in [fla_mat, *trf.T]])
    mats[1] = np.where(mats[1]==0, 1e-4, mats[1])
    mats[3] = np.sqrt(mats[3])
    x = lnr[np.isnan(raw)]
    if x.size > 0:
        upt[np.isnan(upt)] = minimize(
            loss_pdist, x, args=(raw, mats),
            method="L-BFGS-B", jac=jac_pdist, 
            options={"disp":False}, 
        ).x
        # set bins with no available target dists to NaN
        target_nan = np.sum(np.isnan(mats[0]), axis=1)
        upt[target_nan == mats[0].shape[0], :] = np.nan
    return upt