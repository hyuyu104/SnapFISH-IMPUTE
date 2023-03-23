import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as multi

class SnapFISH:
    def __init__(self, coor_path, ann_path, path, suf, w=True):
        self.path = path
        self.suf = suf
        self.write = w
        self.id_col = "cell_id"
        self.pos_col = "pos"

        self.data = pd.read_csv(coor_path, sep="\t")
        ann = pd.read_csv(ann_path, sep="\t")
        self.ann = ann.rename(
            {"Start":"start", "End":"end"}, axis=1
        )

        self.BINSIZE = self.ann["end"][0] - self.ann["start"][0]


    def run_SnapFISH(self, num_nbrs=None, paired=False, singleton=False):
        self.SnapFISH_step1()
        self.SnapFISH_step2(num_nbrs, paired)
        self.SnapFISH_step3()
        self.SnapFISH_step4(singleton)


    def apply_dist(self, x):
        diff = np.array([*combinations(x[["x", "y", "z"]].values, 2)])
        return np.sqrt(np.sum(np.square(diff[:,0,:] - diff[:,1,:]), axis=1))


    def SnapFISH_step1(self):
        CELL_IDS = np.unique(self.data[self.id_col])
        NUM_POS = self.ann.shape[0]

        cell_ids = np.repeat(CELL_IDS, NUM_POS)
        pos_ids = np.tile(self.ann[self.pos_col], len(CELL_IDS))
        full_rids = set(zip(cell_ids, pos_ids))
        raw_rids = set(zip(self.data[self.id_col], self.data[self.pos_col]))
        missed_rows = full_rids - raw_rids
        
        if len(missed_rows) != 0:
            missed_rows = np.array(list(missed_rows))
            missed_df = pd.DataFrame(missed_rows, columns=[self.id_col, self.pos_col])
            self.data = pd.concat([self.data, missed_df])
        self.data = self.data.sort_values([self.id_col, self.pos_col])
        self.data = self.data.reset_index(drop=True)

        dists = self.data.groupby(self.id_col).apply(self.apply_dist)
        dist_vals = np.array(dists.values.tolist())
        self.final = pd.DataFrame(
            dist_vals.T, 
            columns=dists.index
        )
        bins = np.array([*combinations(self.ann[["pos", "start"]].values, 2)])
        self.bin_cols = pd.DataFrame(
            bins.transpose(0, 2, 1).reshape((-1, 4)), 
            columns=["bin1", "bin2", "end1", "end2"]
        )

        self.out_3D_dist = pd.concat([self.bin_cols, self.final], axis=1)
        if not self.write:
            return self.out_3D_dist
        out_path = os.path.join(self.path, f"output_3D_dist_{self.suf}.txt")
        self.out_3D_dist.round(4).to_csv(out_path, sep="\t", index=False)

        mean_dist = self.final.mean(axis=1, skipna=True).to_frame("out.mean")
        out_3D_dist_mean = pd.concat([self.bin_cols, mean_dist], axis=1)
        mean_path = os.path.join(self.path, f"output_3D_dist_{self.suf}_avg.txt")
        out_3D_dist_mean.round(4).to_csv(mean_path, sep="\t", index=False)


    def SnapFISH_step2(self, num_nbrs=None, paired=False):
        CUT_UP, CUT_LO = 50e3, 25e3
        BOUND_LO, BOUND_UP = 1e5, 1e6

        t_test_ls = []
        for i in range(self.out_3D_dist.shape[0]):
            v = self.final.iloc[i,:].dropna()

            b1 = self.bin_cols["end1"][i]
            bin1 = abs(self.bin_cols["end1"] - b1)
            b2 = self.bin_cols["end2"][i]
            bin2 = abs(self.bin_cols["end2"] - b2)

            filter_x = (bin1 <= CUT_UP)&(bin2 <= CUT_UP)&~((bin1 < CUT_LO)&(bin2 < CUT_LO))
            x_bins = self.bin_cols[filter_x]
            
            # local background
            x_vals = self.final[filter_x]
            x_mean = x_vals.mean().dropna()
            if len(x_mean) == 0:
                continue

            xll_vals = x_vals[(x_bins["end1"] > b1)&(x_bins["end2"] < b2)]
            xll = xll_vals.mean().dropna() # lower left

            xh_vals = x_vals[x_bins["end1"] == b1]
            xh = xh_vals.mean().dropna() # horizontal

            xv_vals = x_vals[x_bins["end2"] == b2]
            xv = xv_vals.mean().dropna() # vertical

            if not paired:
                t_stat, p_val = stats.ttest_ind(v, x_mean, equal_var=False)
            else:
                t_stat, p_val = stats.ttest_rel(v, x_mean)
            mean_vals = list(map(
                lambda x: np.nanmean(x) if len(x) != 0 else np.nan, 
                [v, x_mean, xll, xh, xv]
            ))
            bins = self.bin_cols.iloc[i]

            genome_dist = abs(bins["end1"]-bins["end2"])
            bin_bool = genome_dist <= BOUND_LO or genome_dist >= BOUND_UP
            nbr_bool = num_nbrs != None and len(x_vals) != num_nbrs
            v_x_bool = len(v) < 2 or len(x_mean) < 2
            if bin_bool or v_x_bool or nbr_bool:
                continue

            t_test_ls.append(bins.tolist() + mean_vals + [t_stat, p_val])

        bin_col_names = self.bin_cols.columns.tolist()
        t_cols = bin_col_names + \
            ["Case", "Ctrl", "Ctrl.ll", "Ctrl.h", "Ctrl.v", "Tstat", "Pvalue"]
        t_test_df = pd.DataFrame(t_test_ls, columns=t_cols)
        self.test = t_test_df
        pval_corr = multi.multipletests(t_test_df['Pvalue'], method='fdr_bh')[1]
        t_test_df["fdr"] = pval_corr

        self.out_Ttest = t_test_df[
            bin_col_names + \
                ["Case", "Ctrl", "Tstat", "Pvalue", "Ctrl.ll", "Ctrl.h", "Ctrl.v", "fdr"]
        ]
        if not self.write:
            return self.out_Ttest
        Ttest_path = os.path.join(self.path, f"output_Ttest_{self.suf}.txt")
        self.out_Ttest.to_csv(Ttest_path, index=False, sep="\t")


    def SnapFISH_step3(self):
        # only consider bin pairs 100Kb ~ 1MB
        # user defined cutoff
        cutoff1, cutoff2 = 1.1, 1.05

        a = self.out_Ttest.dropna(
            axis = 0,
            subset=["Ctrl", "Ctrl.ll", "Ctrl.h", "Ctrl.v"]
        )

        a["ratio"] = a["Ctrl"] / a["Case"]
        a["ratio.ll"] = a["Ctrl.ll"] / a["Case"]
        a["ratio.h"] = a["Ctrl.h"] / a["Case"]
        a["ratio.v"] = a["Ctrl.v"] / a["Case"]

        x = a[
            (a["Tstat"] < -4) & 
            (a["fdr"] < 0.1) & 
            (a["ratio"] > cutoff1) & 
            (a["ratio.ll"] > cutoff2) & 
            (a["ratio.h"] > cutoff2) & 
            (a["ratio.v"] > cutoff2)
        ]

        x = x.sort_values("fdr", ignore_index=True)

        bins = x[["bin1", "bin2"]].values.flatten()
        bin_locs = self.ann.set_index(self.pos_col)
        bin_locs = bin_locs.loc[bins, ["chr", "start", "end"]]
        bin_locs = bin_locs.values.reshape((-1, 6))
        rec = pd.DataFrame(
            bin_locs, 
            columns = ["chr1","x1","x2","chr2","y1","y2"]
        )

        self.out_candidate = pd.concat([rec, x], axis=1)
        if not self.write:
            return self.out_candidate
        candidate_path = os.path.join(self.path, f"output_loop_candidate_{self.suf}.txt")
        self.out_candidate.to_csv(candidate_path, index=False, sep="\t")


    def spread_label(self, row, df):
        GAP = self.BINSIZE * 2
        neighbors = df[(abs(df['x1']-row['x1'])<=GAP) & (abs(df['y1']-row['y1'])<=GAP)]
        nan_row_idx = neighbors[pd.isna(neighbors["label"])].index
        if len(nan_row_idx) != 0:
            df.loc[nan_row_idx, "label"] = row["label"]
            for _, row in df.loc[nan_row_idx].iterrows():
                self.spread_label(row, df)


    def apply_cluster(self, x):
        x["NegLog10FDR"] = sum(-np.log10(x["fdr"]))
        x["ClusterSize"] = len(x)
        return x.iloc[[np.argmin(x["Tstat"])]]


    def SnapFISH_step4(self, singleton=False):
        result = []
        for _, df in self.out_candidate.groupby("chr1"):
            df["label"] = np.nan
            while df["label"].isna().any():
                na_rows = df.index[pd.isna(df["label"])]
                if len(na_rows) == len(df):
                    df.loc[na_rows[0], "label"] = 1
                else:
                    df.loc[na_rows[0], "label"] = np.max(df["label"]) + 1
                self.spread_label(df.loc[na_rows[0]], df)
            s_df = df.groupby("label", group_keys=False).apply(self.apply_cluster)
            if singleton:
                result.append(s_df)
            else:
                result.append(s_df[s_df["ClusterSize"]>1])
        summit_cols = self.out_candidate.columns.tolist()+["NegLog10FDR", "ClusterSize"]
        self.out_summit = pd.concat(result) if len(result) != 0 \
            else pd.DataFrame(columns=summit_cols)

        summit_path = os.path.join(self.path, f"output_loop_summit_{self.suf}.txt")
        if not self.write:
            return self.out_summit
        self.out_summit.to_csv(summit_path, index=False, sep="\t")