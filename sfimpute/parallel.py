import os
import pandas as pd
import numpy as np
from .impute import *


def generate_single_reg_parallel(MPI, reg_coor_data, reg_ann_df, reg_id, target_fpath):
    """impute the normalized pairwise distances of an imaging region.

    Args:
        MPI (MPI): MPI object.
        reg_coor_data (pd.DataFrame): 3D coordinates of the region.
        reg_ann_df (pd.DataFrame): annotation df of the region.
        reg_id (str): imaging region ID.
        target_fpath (fstr): path of the imputed distances.
    """
    comm = MPI.COMM_WORLD
    dist_df = to_dist_df(reg_coor_data, reg_ann_df)
    reg_norm_df = normalize_pdist_by1d(dist_df)
    reg_pred_df = reg_norm_df.copy()
    pdist_arr = reg_pred_df.to_darr().T

    prev_na, curr_na = None, count_na_in_pdist(pdist_arr)
    if comm.rank == 0:
        print(f"Region {reg_id} initial: {curr_na} NaN values")
    while prev_na != curr_na and curr_na != 0:
        mat_arr = to_mats_single_reg(pdist_arr)

        r = int(np.floor(reg_ann_df["pos"].max() / 20))
        if comm.rank == 0:
            print(f"region {reg_id} resized by a factor of", r)

        resized = np.stack([conv_resize(t, r) for t in mat_arr])
        resized = to_vecs_single_reg(resized)
        dissimil_mat = simil_mat_parallel(resized, MPI)

        P = to_target_parallel(pdist_arr, dissimil_mat, MPI)
        pdist_arr = np.where(np.isnan(pdist_arr), P, pdist_arr)
        prev_na = curr_na
        curr_na = count_na_in_pdist(pdist_arr)
        if comm.rank == 0:
            print(f"Region {reg_id}: {curr_na} NaN values")

    if comm.rank == 0:
        save_path = target_fpath.format(reg_id)
        reg_pred_df[reg_pred_df.vcol()] = pdist_arr.T
        reg_pred_df.round(6).to_csv(save_path, sep="\t", index=False)
    comm.Barrier()


def parallel_dist_mat(
    MPI,
    output_dire,
    suf,
    coor_wnan_path,
    ann_path,
):
    """impute normalize distances of all imaging regions.

    Args:
        MPI (MPI): MPI object.
        output_dire (str): output directory.
        suf (str): file suffix.
        coor_wnan_path (str): 3D coordinate path.
        ann_path (str): annotation file path.

    Returns:
        fstr: pairwise distance path.
    """
    comm = MPI.COMM_WORLD
    coor_data = read_data(coor_wnan_path)
    ann_df = read_data(ann_path)

    target_fpath = "target_dist_" + suf + "_reg{}.txt"
    target_fpath = os.path.join(output_dire, target_fpath)

    np.seterr(divide="ignore", invalid="ignore")
    for reg_id in pd.unique(coor_data["chr"]):
        generate_single_reg_parallel(
            reg_coor_data=coor_data[coor_data["chr"] == reg_id],
            reg_ann_df=ann_df[ann_df["chr"] == reg_id],
            reg_id=reg_id,
            target_fpath=target_fpath,
            MPI=MPI,
        )
    comm.Barrier()  # ensures that all targets are generated
    np.seterr(divide="warn", invalid="warn")

    return target_fpath


def parallel_single_recover(
    MPI, coor_wnan_path, lnr_path, reg_target_path, reg_id, recover_fpath
):
    """recover the 3D coordinates of a single genomic region by distributing chromosomes to
    different processes and run parallelly.

    Args:
        MPI (MPI): MPI object.
        coor_wnan_path (str): path to the 3D coordinates, missing values are replaced by NaN.
        lnr_path (str): (N*K)*3, path to linear imputed 3D coordinates,
        reg_target_path (str): output of parallel_target.
        reg_id (str): the ID of the region to recover.
        recover_fpath (fstr): output path to save the recovered 3D coordinates.
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        coorw_df = read_data(coor_wnan_path)
        lnr_df = read_data(lnr_path)
        coor_reg = coorw_df[coorw_df["chr"] == reg_id]
        lnr_reg = lnr_df[lnr_df["chr"] == reg_id]

        tar_chr = DistDataFrame(read_data(reg_target_path))

        assert np.all(pd.unique(coor_reg["cell_id"]) == pd.unique(lnr_reg["cell_id"]))

        coorbuff = np.stack(
            coor_reg.groupby("cell_id", sort=False)
            .apply(lambda x: x.sort_values("pos")[["x", "y", "z"]].values)
            .values
        )
        lnrbuff = np.stack(
            lnr_reg.groupby("cell_id", sort=False)
            .apply(lambda x: x.sort_values("pos")[["x", "y", "z"]].values)
            .values
        ).ravel(order="C")
        tarbuff = tar_chr[pd.unique(coor_reg["cell_id"])].values.T

        shapes = np.array([coorbuff.shape[1], tarbuff.shape[1]])

        ave, res = divmod(coorbuff.shape[0], comm.size)
        count = np.array([ave + 1 if p < res else ave for p in range(comm.size)])
        index = np.array([sum(count[:p]) for p in range(comm.size)])
        index_a, index_b = index * shapes[0] * 3, index * shapes[1]

        coorbuff = coorbuff.ravel(order="C")
        tarbuff = tarbuff.ravel(order="C")
    else:
        coorbuff, lnrbuff, tarbuff = None, None, None
        count = np.zeros(comm.size, dtype="int")
        index_a, index_b = None, None
        shapes = np.zeros(2, dtype="int")

    comm.Bcast(count, root=0)
    comm.Bcast(shapes, root=0)

    if comm.rank == 0:
        trf = tar_chr[["lmbda", "mu", "var"]].values.ravel(order="C")
    else:
        trf = np.zeros(shapes[1] * 3, dtype="float")
    comm.Bcast(trf, root=0)
    trf = trf.reshape((shapes[1], 3))

    count_a = count * shapes[0] * 3
    coorrec = np.zeros(count_a[comm.rank], dtype="float")
    comm.Scatterv([coorbuff, count_a, index_a, MPI.DOUBLE], coorrec, root=0)
    coorrec = coorrec.reshape((count[comm.rank], shapes[0], 3))

    lnrrec = np.zeros(count[comm.rank] * shapes[0] * 3, dtype="float")
    comm.Scatterv([lnrbuff, count_a, index_a, MPI.DOUBLE], lnrrec, root=0)
    lnrrec = lnrrec.reshape((count[comm.rank], shapes[0], 3))

    count_b = count * shapes[1]
    tarrec = np.zeros([count[comm.rank], shapes[1]], dtype="float")
    comm.Scatterv([tarbuff, count_b, index_b, MPI.DOUBLE], tarrec, root=0)
    tarrec = tarrec.reshape((count[comm.rank], shapes[1]))

    opt = np.array(
        [
            recover_3d_coor_single_chr(raw, lnr, fla_mat, trf)
            for raw, lnr, fla_mat in zip(coorrec, lnrrec, tarrec)
        ]
    ).ravel(order="C")

    opt_all = np.zeros(np.sum(count_a), dtype="float")
    comm.Gatherv(opt, [opt_all, count_a, index_a, MPI.DOUBLE], root=0)
    opt_all = opt_all.reshape((-1, shapes[0], 3))

    if comm.rank == 0:
        opt_chr_df = coor_reg.copy()
        opt_chr_df[["x", "y", "z"]] = opt_all.reshape((-1, 3))
        save_coor(opt_chr_df, coor_wnan_path, recover_fpath.format(reg_id))


def parallel_wrapper(MPI, output_dire, suf, ann_path, coor_wnan_path):
    """impute missing 3D coordinates parallelly.

    Args:
        MPI (MPI): MPI object.
        output_dire (str): the directory to store output files.
        suf (str): file name suffix.
        ann_path (str): path of the annotation file.
        coor_wnan_path (str): path of 3D coordinates.
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0 and not os.path.exists(output_dire):
        os.mkdir(output_dire)
    comm.Barrier()

    target_fpath = parallel_dist_mat(
        MPI=MPI,
        output_dire=output_dire,
        suf=suf,
        coor_wnan_path=coor_wnan_path,
        ann_path=ann_path,
    )

    # target_fpath = os.path.join(output_dire, "target_dist_" + suf + "_reg{}.txt")

    lnr_path = os.path.join(output_dire, f"linear_coor_{suf}.txt")
    if comm.rank == 0 and not os.path.exists(lnr_path):
        generate_interpolation(ann_path, coor_wnan_path, "linear", lnr_path)
    comm.Barrier()

    coorw_df = read_data(coor_wnan_path)
    recover_fpath = "recover_coor_" + suf + "_reg{}.txt"
    recover_fpath = os.path.join(output_dire, recover_fpath)

    for reg_id in pd.unique(coorw_df["chr"]):
        parallel_single_recover(
            MPI=MPI,
            reg_id=reg_id,
            recover_fpath=recover_fpath,
            coor_wnan_path=coor_wnan_path,
            lnr_path=lnr_path,
            reg_target_path=target_fpath.format(reg_id),
        )

    all_coor_path = os.path.join(output_dire, f"recover_coor_{suf}.txt")
    if comm.rank == 0:
        recover_coors = pd.concat(
            [
                read_data(recover_fpath.format(reg_id))
                for reg_id in pd.unique(coorw_df["chr"])
            ],
            ignore_index=True,
            sort=False,
        )
        save_coor(recover_coors, coor_wnan_path, all_coor_path)
        for reg_id in pd.unique(coorw_df["chr"]):
            os.remove(target_fpath.format(reg_id))
            os.remove(recover_fpath.format(reg_id))
        print("Ended")
    comm.Barrier()
