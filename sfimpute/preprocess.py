import os, re
import pandas as pd
import numpy as np
from .impute import read_data


def process_mESC_seqFISH_25kb(
        mESC_dire="data_mESC_seqFISH", 
        mBCC_dire="data_mBCC_seqFISH", 
):
    """process the 25Mb subset from the mESCs dataset. 

    Args:
        mESC_dire (str): mESCs data directory.
        mBCC_dire (str): mouse brain cells data directory.

    Returns:
        str, str: processed 3D coordinates path, processed 1D annotation file path
    """
    resol = "25kb"
    mESC_fpath = "DNAseqFISH+{}loci-E14-replicate{}.csv"

    # the Nature paper has the same barcoding
    ann_path = os.path.join(mBCC_dire, "science.abj1966_table_s1.xlsx")
    ann = pd.read_excel(ann_path, sheet_name="25-kb resolution").dropna()
    ann_processed = mBCC_process_ann(ann)

    # mapping series, convert geneID to locus ID
    conv_sr = ann_processed["pos"].copy()
    conv_sr.index = ann["geneID"]
    # mapping series, convert geneID to region ID
    chr_sr = ann_processed["chr"].copy()
    chr_sr.index = ann["geneID"]

    rep1_data = pd.read_csv(os.path.join(mESC_dire, mESC_fpath.format(resol, 1)))
    rep1_data.insert(0, "rep", 1)
    rep2_data = pd.read_csv(os.path.join(mESC_dire, mESC_fpath.format(resol, 2)))
    rep2_data.insert(0, "rep", 2)
    # merge data from the two biological replicates
    mESC_data = pd.concat([rep1_data, rep2_data], sort=False, ignore_index=True)

    # labelID -> which haploid each locus belongs to
    mESC_data = mESC_data[(mESC_data["labelID"]==0)|(mESC_data["labelID"]==1)]
    mESC_data = mESC_data.rename({"regionID (hyb1-60)":"geneID"}, axis=1)
    mESC_data = mESC_data.reset_index(drop=True)

    grp_cols = ["rep", "fov", "cellID", "labelID"]
    mESC_data["chr"] = mESC_data["chromID"]
    mESC_data["pos"] = mESC_data["geneID"]
    mESC_data = mESC_data.sort_values(
        grp_cols[:2] + ["chr"] + grp_cols[2:] + ["pos", "dot_intensity"]
    ).reset_index(drop=True)

    mESC_data_filtered = mESC_data.groupby(
        grp_cols[:2] + ["chr"] + grp_cols[2:] + ["pos"], 
        sort=False
    ).tail(1).reset_index(drop=True)

    c = grp_cols[0]
    chr_id = c + "." + mESC_data_filtered[c].astype("str")
    for c in grp_cols[1:]:
        chr_id += f".{c}." + mESC_data_filtered[c].astype("str")
    mESC_data_filtered["cell_id"] = chr_id
    mESC_data_filtered = mESC_data_filtered[
        ["chr", "cell_id", "pos", "x", "y", "z"]
    ]

    # convert to nm
    mESC_data_filtered["x"] = (mESC_data_filtered["x"] * 103).round(6)
    mESC_data_filtered["y"] = (mESC_data_filtered["y"] * 103).round(6)
    mESC_data_filtered["z"] = (mESC_data_filtered["z"] * 250).round(6)

    print(
        f"mESC seqFISH+ {resol} rep1&rep2:",
        len(pd.unique(mESC_data_filtered["cell_id"])), 
        "chromosomes"
    ) # -> 892 chromosomes in total
    coor_path = os.path.join(mESC_dire, f"mESC_seqFISH_{resol}_coor.txt")
    mESC_data_filtered.to_csv(coor_path, index=False, sep="\t")
    ann_path = os.path.join(mESC_dire, f"mESC_seqFISH_{resol}_ann.txt")
    ann_processed.to_csv(ann_path, index=False, sep="\t")
    return coor_path, ann_path


def process_mESC_seqFISH_1Mb(jie_dire, mBCC_dire, mESC_dire):
    """process the 1Mb subset from the mESCs dataset. 

    Args:
        jie_dire (str): jie aligned result directory.
        mBCC_dire (str): mouse brain cells data directory.
        mESC_dire (str): mESCs data directory.

    Returns:
        str, str: processed 3D coordinates path, processed 1D annotation file path
    """
    data = []
    if "mESC" in jie_dire and "1Mb" in jie_dire:
        ids4DN = ["FIS6MLXGA", "FIU73OR5W"]
        output_dire = mESC_dire
    elif "mBCC" in jie_dire and "1Mb" in jie_dire:
        ids4DN = ["FIYNWVJEP", "FIFBXKXK9", "FIXGTJBGU"]
        output_dire = mBCC_dire
    for i, id4DN in enumerate(ids4DN):
        data_path = os.path.join(jie_dire, f"4DN{id4DN}.csv")
        info_lines = []
        with open(data_path, "r") as f:
            line = f.readline()
            while line.startswith("#") or line.startswith('"'):
                info_lines.append(line)
                line = f.readline()
        col_str = re.sub("^.*\((.*)\).*$", "\g<1>", info_lines[-1]).strip().lower()
        d = pd.read_csv(data_path, skiprows=len(info_lines), names=col_str.split(","))
        d["rep"] = i + 1
        data.append(d)
    data = pd.concat(data, sort=False, ignore_index=True)

    id_cols = ["rep", "fov", "cellID", "chr", "labelID"]
    data[id_cols[1:]] = np.stack(data["trace_id"].str.split("_").values).astype("int")

    pos_raw = data["chr"].astype("str") + "." + data["chrom_start"].astype("str")

    ann_path = os.path.join(mBCC_dire, "science.abj1966_table_s1.xlsx")
    ann = pd.read_excel(ann_path, sheet_name="1-Mb resolution").dropna()
    ann_processed = mBCC_process_ann(ann)

    pos_map = ann_processed["chr"].astype("str") + "." + ann_processed["start"].astype("str")
    pos_map = pd.Series(ann_processed["pos"].values, index=pos_map.values)
    data["pos"] = pos_raw.map(pos_map)

    sort_cols = ["chr", "rep", "fov", "cellID", "labelID", "pos"]
    data = data.sort_values(sort_cols, ignore_index=True)

    grp_cols = ["rep", "fov", "cellID", "labelID"]
    c = grp_cols[0]
    chr_id = c + "." + data[c].astype("str")
    for c in grp_cols[1:]:
        chr_id += f".{c}." + data[c].astype("str")
    data["cell_id"] = chr_id
    data = data[["chr", "cell_id", "pos", "x", "y", "z"]]

    data["x"] = (data["x"]*1000).round(6)
    data["y"] = (data["y"]*1000).round(6)
    data["z"] = (data["z"]*1000).round(6)

    data_name = jie_dire.split("/")[-1]
    coor_path = os.path.join(output_dire, f"{data_name}_coor.txt")
    data.to_csv(coor_path, index=False, sep="\t")
    ann_path = os.path.join(output_dire, f"{data_name}_ann.txt")
    ann_processed.to_csv(ann_path, index=False, sep="\t")
    return coor_path, ann_path


def process_mBCC_seqFISH_1Mb(mBCC_dire="data_mBCC_seqFISH"):
    """process the 1Mb subset from the mouse brain cell dataset. 

    Args:
        mBCC_dire (str, optional): mouse brain cell data direcctory. Defaults to "data_mBCC_seqFISH".

    Returns:
        str, str: processed 3D coordinates path, processed 1D annotation file path
    """
    mBCC_dire = "data_mBCC_seqFISH"
    coor_path = os.path.join(mBCC_dire, "TableS7_brain_DNAseqFISH_1Mb_voxel_coordinates_2762cells.csv")
    coor_data = pd.read_csv(coor_path)

    coor_data = coor_data[coor_data["labelID"] >= 0]
    coor_data["x"] = (coor_data["x"]*103).round(6)
    coor_data["y"] = (coor_data["y"]*103).round(6)
    coor_data["z"] = (coor_data["z"]*250).round(6)

    ann_path = os.path.join(mBCC_dire, "science.abj1966_table_s1.xlsx")
    ann = pd.read_excel(ann_path, sheet_name="1-Mb resolution").dropna()
    ann_processed = mBCC_process_ann(ann)

    pos = np.concatenate(ann.groupby("chromID", sort=False).apply(
        lambda x: np.arange(1, len(x)+1)
    ).values)
    pos_map = pd.Series(pos, index=ann["geneID"])
    coor_data["pos"] = coor_data["geneID"].map(pos_map)
    coor_data = coor_data.drop("rep", axis=1).rename({
        "replicateID":"rep", "cluster label":"cluster", "chromID":"chr"
    }, axis=1)

    grp_cols = ["rep", "fov", "cellID", "labelID"]
    kept_cols = grp_cols + ["chr", "pos"] + ["x", "y", "z"] + ["cluster"]
    sub_cols = coor_data[kept_cols]
    kept_rows = sub_cols.groupby(
        kept_cols[:-4], sort=False
    ).head(1).reset_index(drop=True)

    sort_cols = ["chr", "rep", "fov", "cellID", "labelID", "pos"]
    kept_rows = kept_rows.sort_values(sort_cols, ignore_index=True)

    c = grp_cols[0]
    chr_id = c + "." + kept_rows[c].astype("str")
    for c in grp_cols[1:]:
        chr_id += f".{c}." + kept_rows[c].astype("str")
    kept_rows["cell_id"] = chr_id
    data = kept_rows[["chr", "cell_id", "pos", "x", "y", "z", "cluster"]]

    coor_path = os.path.join(mBCC_dire, "mBCC_seqFISH_1Mb_coor.txt")
    data.to_csv(coor_path, index=False, sep="\t")
    ann_path = os.path.join(mBCC_dire, "mBCC_seqFISH_1Mb_ann.txt")
    ann_processed.to_csv(ann_path, index=False, sep="\t")
    return coor_path, ann_path


def insert_nan_to_coor(coor_path, ann_path):
    """insert NaN to raw 3D coordinates.

    Args:
        coor_path (str): the processed 3D coordinates file path.
        ann_path (str): the processed 1D genomic location annotation file path.
    """
    data = read_data(coor_path)
    ann = read_data(ann_path)

    inserted_nan = []
    for reg_id, df in data.groupby("chr", sort=False):

        sub_ann = ann[ann["chr"]==reg_id]
        chr_ids = pd.unique(df["cell_id"])

        num_loci, num_chrs = len(sub_ann), len(chr_ids)
        full_chr_ids = np.repeat(chr_ids, num_loci)
        full_pos = np.tile(sub_ann["pos"], num_chrs)

        full_df = pd.DataFrame({"chr":[reg_id]*len(full_pos), "cell_id":full_chr_ids, "pos":full_pos})
        inserted_nan.append(full_df.merge(df, how="left", on=["chr", "cell_id", "pos"], sort=False))

    pd.concat(inserted_nan, sort=False, ignore_index=True).to_csv(
        coor_path[:-4] + "_wnan" + coor_path[-4:],
        index=False, sep="\t"
    )


def mBCC_process_ann(ann):
    """process the 1D annotation file from Takei et al.

    Args:
        ann (pd.DataFrame): raw annotation file.

    Returns:
        pd.DataFrame: chr, pos, start, end
    """
    ann = ann.astype({"chromID":"int", "Start":"int", "End":"int"})

    ann["pos"] = ann.groupby("chromID", sort=False).apply(
        lambda x: x["regionID"] - np.min(x["regionID"]) + 1
    ).values.astype("int")
    return ann[["chromID", "pos", "Start", "End"]].rename({
        "chromID":"chr", "Start":"start", "End":"end"
    }, axis=1)


def join_sox2(sox2_dire="data_mESC_Sox2"):
    """merge the two alleles in the Sox2 dataset.

    Args:
        sox2_dire (str, optional): Sox2 data directory. Defaults to "data_mESC_Sox2".
    """
    al129 = read_data(os.path.join(sox2_dire, "allele_129_coor_wnan.txt"))
    alcast = read_data(os.path.join(sox2_dire, "allele_cast_coor_wnan.txt"))
    al129["chr"], alcast["chr"] = "129", "cast"
    alall = pd.concat([al129, alcast], axis=0, sort=False, ignore_index=True)
    alall.round(6).to_csv(
        os.path.join(sox2_dire, "mESC_Sox2_coor_wnan.txt"), sep="\t", index=False
    )

    ann = read_data(os.path.join(sox2_dire, "sox2_ann.txt"))
    ann129, anncast = ann.copy(), ann.copy()
    ann129["chr"], anncast["chr"] = "129", "cast"
    annall = pd.concat([ann129, anncast], axis=0, sort=False, ignore_index=True)
    annall.to_csv(os.path.join(sox2_dire, "mESC_Sox2_ann.txt"), sep="\t", index=False)


def process_all(
        mESC_dire="data_mESC_seqFISH", 
        mBCC_dire="data_mBCC_seqFISH", 
        jie_dire="jie_aligned"
):
    """process mESCs seqFISH+ 25kb&1Mb and mouse brain cells seqFISH+ 1Mb.

    Args:
        mESC_dire (str, optional): mESCs data directory. Defaults to "data_mESC_seqFISH".
        mBCC_dire (str, optional): mouse brain cells data directory. Defaults to "data_mBCC_seqFISH".
        jie_dire (str, optional): jie aligned result directory. Defaults to "jie_aligned".
    """
    coor_path, ann_path = process_mESC_seqFISH_25kb(mESC_dire=mESC_dire, mBCC_dire=mBCC_dire)
    insert_nan_to_coor(coor_path=coor_path, ann_path=ann_path)

    jie_subdire = os.path.join(jie_dire, "mESC_seqFISH_1Mb")
    coor_path, ann_path = process_mESC_seqFISH_1Mb(jie_subdire, mBCC_dire, mESC_dire)
    insert_nan_to_coor(coor_path, ann_path)

    jie_subdire = os.path.join(jie_dire, "mBCC_seqFISH_1Mb")
    coor_path, ann_path = process_mESC_seqFISH_1Mb(jie_subdire, mBCC_dire, mESC_dire)
    insert_nan_to_coor(coor_path, ann_path)

    coor_path, ann_path = process_mBCC_seqFISH_1Mb()
    insert_nan_to_coor(coor_path, ann_path)

    mbcc_wnan_coor_path = "data_mBCC_seqFISH/mBCC_seqFISH_1Mb_coor_wnan.txt"
    data = pd.read_csv(mbcc_wnan_coor_path, sep="\t")
    cell_cluster = data.dropna()[["cell_id", "cluster"]].drop_duplicates()
    cluster_map = pd.Series(cell_cluster["cluster"].values, index=cell_cluster["cell_id"])
    data["cluster"] = data["cell_id"].map(cluster_map).astype("int")
    data.round(6).to_csv(mbcc_wnan_coor_path, sep="\t", index=False)