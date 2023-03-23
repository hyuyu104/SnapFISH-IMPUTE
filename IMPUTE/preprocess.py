import os
import pandas as pd
import numpy as np

def preprocess_mESC_sox2():
    in_dire = os.path.join("data", "mESC_sox2_raw")
    out_dire = os.path.join("data", "mESC_sox2_nm")
    if not os.path.exists(out_dire):
        os.mkdir(out_dire)
    cast_path = os.path.join(in_dire, "060422_CASTalleles_649cells_18005rows.txt")
    pd.read_csv(
        cast_path, sep="\t"
    ).rename(
        {"hyb":"pos"}, axis=1
    )[["allele", "cell_id", "pos", "x", "y", "z"]].to_csv(
        os.path.join(out_dire, "mESC_CAST_allele.txt"),
        sep="\t", index=False
    )
    a129_path = os.path.join(in_dire, "060322_129alleles_649cells_18189rows.txt")
    pd.read_csv(
        a129_path, sep="\t"
    ).rename(
        {"hyb":"pos"}, axis=1
    )[["allele", "cell_id", "pos", "x", "y", "z"]].to_csv(
        os.path.join(out_dire, "mESC_129_allele.txt"),
        sep="\t", index=False
    )
    pd.read_csv(
        os.path.join(in_dire, "input_ann.txt"), sep="\t"
    ).to_csv(
        os.path.join(out_dire, "input_ann.txt"),
        sep="\t", index=False
    )


def preprocess_mESC_seqFISH():
    in_dire = os.path.join("data", "mESC_seqFISH+_raw")
    rep1_path = os.path.join(in_dire, "mESC_DNAseqFISH+_25Kb_autosomes_Rep1.txt")
    rep1 = pd.read_csv(rep1_path, sep="\t")
    rep2_path = os.path.join(in_dire, "mESC_DNAseqFISH+_25Kb_autosomes_Rep2.txt")
    rep2 = pd.read_csv(rep2_path, sep="\t")

    rep1["cell_id"] = "chrom." + rep1["chromID"].astype("str") + \
        "." + rep1["alleleNAME"].astype("str")
    n1 = np.unique(rep1["cell_id"])
    n1_dict = dict(np.array([n1, np.arange(0, len(n1))]).T)
    rep1["cell_id"] = rep1["cell_id"].map(n1_dict)
    rep2["cell_id"] = "chrom." + rep2["chromID"].astype("str") + \
        "." + rep2["alleleNAME"].astype("str")
    n2 = np.unique(rep2["cell_id"])
    n2_dict = dict(np.array([n2, np.arange(len(n1), len(n1)+len(n2))]).T)
    rep2["cell_id"] = rep2["cell_id"].map(n2_dict)

    data = pd.concat([rep1, rep2]).sort_values(["cell_id", "regionID..hyb1.60."])
    data = data.rename({"regionID..hyb1.60.":"pos"}, axis=1)
    data = data[["chromID", "cell_id", "pos", "x", "y", "z"]]
    data = data[data["chromID"].isin(np.arange(1, 20))]

    # convert to nm
    data["x"] = data["x"] * 103
    data["y"] = data["y"] * 103
    data["z"] = data["z"] * 250

    out_dire = os.path.join("data", "mESC_seqFISH+_nm")
    if not os.path.exists(out_dire):
        os.mkdir(out_dire)

    for chr, df in data.groupby("chromID"):
        df.sort_values(["cell_id", "pos"]).to_csv(
            os.path.join(out_dire, f"mESC_chr{chr}_coors.txt"), 
            sep="\t", index=False
        )

    all_ann = pd.read_csv(
        os.path.join(in_dire, "data_ann_25kb_loci.txt"),
        sep = "\t"
    ).rename(
        {"Start":"start", "End":"end", "Chrom ID":"chr"}, axis=1
    )[["chr", "start", "end"]]
    all_ann = all_ann[all_ann["chr"].isin(np.arange(1, 20))]
    all_ann["pos"] = np.concatenate(
        all_ann.groupby("chr").apply(
            lambda x: np.arange(1, len(x)+1),
        ).values
    )

    for chr, df in all_ann.groupby("chr"):
        df[["chr", "start", "end", "pos"]].to_csv(
            os.path.join(out_dire, f"mESC_chr{chr}_ann.txt"), 
            sep="\t", index=False
        )

def combine_mESC_seqFISH_results():
    lnr_dire = os.path.join("output", "mESC_seqFISH+", "linear_output")
    pd.concat([
        pd.read_csv(
            os.path.join(lnr_dire, f"mESC_lnr_mid_chr{i}_pred.txt"),
            sep="\t"
        ) for i in range(1, 20)
    ]).reset_index(drop=True).to_csv(
        os.path.join("output", "mESC_seqFISH+", "mESC_seqFISH+_linear.txt"),
        sep="\t", index=False
    )

    step1_dire = os.path.join("output", "mESC_seqFISH+", "step1_output")
    pd.concat([
        pd.read_csv(
            os.path.join(step1_dire, f"mESC_mr_update_chr{i}_pred.txt"),
            sep="\t"
        ) for i in range(1, 20)
    ]).reset_index(drop=True).to_csv(
        os.path.join("output", "mESC_seqFISH+", "mESC_seqFISH+_step1.txt"),
        sep="\t", index=False
    )

    step2_dire = os.path.join("output", "mESC_seqFISH+", "step2_output")
    pd.concat([
        pd.read_csv(
            os.path.join(step2_dire, f"mESC_agg_itern_chr{i}_pred.txt"),
            sep="\t"
        ) for i in range(1, 20)
    ]).reset_index(drop=True).to_csv(
        os.path.join("output", "mESC_seqFISH+", "mESC_seqFISH+_step2.txt"),
        sep="\t", index=False
    )

if __name__ == "__main__":
    preprocess_mESC_sox2()
    preprocess_mESC_seqFISH()
    # combine_mESC_seqFISH_results()