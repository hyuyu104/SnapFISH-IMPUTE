import os, re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

class NaiveImputer:
    def __init__(self, ann_path, coor_path):
        self.m, self.chr = "pos", "cell_id"
        self.coors = ["x", "y", "z"]
        self.cols = [self.chr, self.m] + self.coors
        self.ann_path, self.coor_path = ann_path, coor_path
        self.initial_s = 500

        self.preprocessing()

        self.predicted = {"lnr_ext":[], "lnr_mid":[], "spl_raw":[], "spl_opt":[]}
        self.mde_list = {"lnr_ext":[], "lnr_mid":[], "spl_raw":[], "spl_opt":[]}


    def preprocessing(self):
        ann = pd.read_csv(self.ann_path, sep="\t")
        convert_dict = {"hyb":self.m, "Start":"start", "End":"end",
                        "Chrom ID":"chr", "Region ID":self.m}
        self.ann = ann.rename(convert_dict, axis=1)

        data = pd.read_csv(self.coor_path, sep="\t")
        self.data = data.rename(convert_dict, axis=1)[self.cols]
        min_diff = min(self.data[self.m]) - min(self.ann[self.m])
        max_diff = max(self.data[self.m]) - max(self.ann[self.m])
        if min_diff != max_diff:
            raise Exception("tail loci is absent")
        else:
            self.data[self.m] -= min_diff

        ann_dist = (self.ann["start"] - self.ann["start"].shift())[1:]
        ann_dist.index = list(zip(self.ann[self.m][:-1], self.ann[self.m][1:]))
        self.ann = self.ann.set_index(self.m, drop=False)
        self.ann_dist = ann_dist


    def avg_MSE(self, pred_type, tr_porp=0.9):
        mde = []
        for CHR, df in self.data.groupby(self.chr):
            sampled = df.sample(int(len(df)*tr_porp)).sort_values(self.m)

            if pred_type == "lnr_ext":
                pred = self.predict_single_linear_extend(sampled)
            if pred_type == "lnr_mid":
                pred = self.predict_single_linear_middle(sampled)
            if pred_type == "spl_raw":
                pred = self.predict_single_spline_raw(sampled)
            if pred_type == "spl_opt":
                pred = self.predict_single_spline_optims(sampled)

            self.predicted[pred_type].append(pred)

            verify_idx = np.intersect1d(df[self.m], pred[self.m])
            true_vals = df.set_index(self.m).loc[verify_idx, self.coors]
            pred_vals = pred.set_index(self.m).loc[verify_idx, self.coors]
            diff = true_vals.values - pred_vals.values
            mde.extend(np.sqrt(np.sum(np.square(diff), axis=1)))

        self.mde_list[pred_type].append(mde)


    def apply_miss(self, x):
        if x[self.m]>x["miss"]>x[self.m]-3:
            return (x["miss"],x["miss"]-1,x["miss"]+1,0)
        elif x[self.m] == x["miss"]:
            return (x["miss"],x["miss"]-2,x["miss"]-1,1)
        return (x["miss"],x["miss"]+1,x["miss"]+2,-1)


    def miss_neighbors_extend(self, sampled):
        loci_sr = pd.Series([0]*len(self.ann), index=self.ann[self.m])
        loci_sr[sampled[self.m]] = 1

        missing_sr = loci_sr.rolling(4).apply(
            lambda x: x[x==0].index[0] if x.sum()==3 else -1)
        miss_sr = missing_sr[missing_sr>=0].dropna()

        if len(miss_sr) == 0:
            return pd.DataFrame()
        
        miss_df = miss_sr.to_frame("miss").reset_index()

        miss_neigh = miss_df.apply(self.apply_miss, axis=1, result_type="expand")
        miss_neigh.columns = ["miss", "n1", "n2", "label"]
        grp_mis = miss_neigh.drop_duplicates().groupby("miss", group_keys=False)
        f_df = grp_mis.apply(lambda x: x if len(x)==1 else x[x["label"]==0])
        return f_df.set_index("miss")


    def predict_single_linear_extend(self, sampled):
        miss_df = self.miss_neighbors_extend(sampled)
        sampled = sampled.set_index(self.m, drop=True)

        predicted = []
        for miss, row in miss_df.iterrows():

            if row["label"] == 0: # given x1 and x3, impute x2
                sum_dist = ((self.ann_dist[row["n1"], miss])+(self.ann_dist[miss, row["n2"]]))
                r = (self.ann_dist[row["n1"], miss])/sum_dist
                p = (1-r)*sampled.loc[row["n1"],self.coors] + r*sampled.loc[row["n2"],self.coors]

            elif row["label"] == 1: # given x1 and x2, impute x3
                sum_dist = ((self.ann_dist[row["n1"], row["n2"]])+(self.ann_dist[row["n2"], miss]))
                r = (self.ann_dist[row["n1"], row["n2"]])/sum_dist
                p = sampled.loc[row["n2"],self.coors]/r + (r-1)*sampled.loc[row["n1"],self.coors]/r

            elif row["label"] == -1: # given x2 and x3, impute x1
                sum_dist = ((self.ann_dist[miss, row["n1"]])+(self.ann_dist[row["n1"], row["n2"]]))
                r = (self.ann_dist[miss, row["n1"]])/sum_dist
                p = sampled.loc[row["n1"],self.coors]/(1-r) + r*sampled.loc[row["n2"],self.coors]/(r-1)

            predicted.append([sampled[self.chr].iloc[0], miss] + p.tolist())

        return pd.DataFrame(predicted, columns=self.cols)
    

    def miss_ratio(self, ls_in):
        ls = sorted(ls_in)
        sum_dist = self.ann.loc[ls[2]]["start"] - self.ann.loc[ls[0]]["start"]
        diff_dist = self.ann.loc[ls[1]]["start"] - self.ann.loc[ls[0]]["start"]
        return ls_in + [diff_dist/sum_dist]


    def miss_neighbors_middle(self, sampled):
        loci_sr = pd.Series(
            [0]*len(self.ann), 
            index=self.ann[self.m]
        )
        loci_sr[sampled[self.m]] = 1
        p0, p1, q2, r = None, None, deque(), []
        
        for i in loci_sr.index:
            if loci_sr[i] == 0:
                q2.append(i)
            if loci_sr[i] == 1 or i == loci_sr.index[-1]:
                if p1 != None:
                    while len(q2) != 0:
                        if i != loci_sr.index[-1]:
                            r.append(self.miss_ratio([q2.popleft(), p1, i]))
                        else:
                            r.append(self.miss_ratio([q2.popleft(), p0, p1]))
                p0, p1 = p1, i

        return pd.DataFrame(
            r, columns=["miss", "n1", "n2", "r"]
        ).set_index("miss")


    def predict_single_linear_middle(self, sampled):
        miss_df = self.miss_neighbors_middle(sampled)
        sampled = sampled.set_index(self.m, drop=False)

        predicted = []
        for miss, row in miss_df.iterrows():

            if row["n2"] > miss > row["n1"]: # impute x2
                p1 = (1-row["r"])*sampled.loc[row["n1"],self.coors]
                p2 = row["r"]*sampled.loc[row["n2"],self.coors]

            elif row["n2"] < miss: # impute x3
                p1 = sampled.loc[row["n2"],self.coors]/row["r"]
                p2 = (row["r"]-1)*sampled.loc[row["n1"],self.coors]/row["r"]

            elif row["n1"] > miss: # impute x1
                p1 = sampled.loc[row["n1"],self.coors]/(1-row["r"])
                p2 = row["r"]*sampled.loc[row["n2"],self.coors]/(row["r"]-1)

            predicted.append([sampled[self.chr].iloc[0], miss] + (p1+p2).tolist())

        return pd.DataFrame(predicted, columns=self.cols)
    

    def predict_single_spline_raw(self, sampled):
        shift_ann = (self.ann["start"] - self.ann.iloc[0]["start"])
        normal_ann = shift_ann / shift_ann.iloc[-1]

        build_u = normal_ann.loc[sampled[self.m]].values
        miss_pos = np.setdiff1d(normal_ann.index, sampled[self.m])
        pred_u = normal_ann.loc[miss_pos].values
        if len(pred_u) == 0:
            return pd.DataFrame(columns=self.cols)

        tck = splprep(sampled[self.coors].values.T, u=build_u, s=0, k=3)[0]
        pred_df = pd.DataFrame(
            np.stack(splev(pred_u, tck)).T, columns=self.coors)
        pred_df[self.m] = miss_pos
        pred_df[self.chr] = sampled[self.chr].iloc[0]
        return pred_df[self.cols]
    

    def predict_single_spline_optims(self, sampled):
        shift_ann = (self.ann["start"] - self.ann.iloc[0]["start"])
        normal_ann = shift_ann / shift_ann.iloc[-1]

        build_u = normal_ann.loc[sampled[self.m]].values.reshape((-1, 1))
        miss_pos = np.setdiff1d(normal_ann.index, sampled[self.m])
        pred_u = normal_ann.loc[miss_pos]
        if len(pred_u) == 0:
            return pd.DataFrame(columns=self.cols)
        
        build_arr = np.hstack([build_u, sampled[self.coors].values])

        def eval_spl(i, s):
            feed_arr = np.vstack([build_arr[:i],build_arr[i+1:]])
            if s < 0:
                return np.array([[np.inf]])
            out_ = splprep(feed_arr[:,1:].T, u=feed_arr[:,0], k=3, s=s, full_output=1)
            tck, self.flag = out_[0][0], self.flag and out_[2] <= 0
            return splev(build_arr[i,0], tck) - build_arr[i,1:]
        
        def eval_smooth(s):
            m = map(eval_spl, np.arange(0, len(build_arr)), np.repeat(s, len(build_arr)))
            return np.mean(np.sqrt(np.sum(np.square(np.stack(list(m))), axis=1)))

        # self.flag, j = False, 1
        # while not self.flag:
        #     if j >= 2:
        #         print("previous iteration failed, j =", j)
        #     self.flag, init_s, j = True, self.initial_s*j, j+1
        #     optimal_s = minimize(eval_smooth, init_s, method='nelder-mead', 
        #                         options={'xatol': 1e-5, 'disp': False}).x[0]
        
        self.flag = True
        optimal_s = minimize(eval_smooth, self.initial_s, method='nelder-mead', 
                             options={'xatol': 1e-5, 'disp': False}).x[0]
        if not self.flag: # if s is too small, set s to 0
            optimal_s = 0

        tck = splprep(sampled[self.coors].values.T, u=build_u.flatten(), k=3, s=optimal_s)[0]
        pred_df = pd.DataFrame(np.stack(splev(pred_u, tck)).T, columns=self.coors)
        pred_df[self.m], pred_df[self.chr] = miss_pos, sampled[self.chr].iloc[0]
        return pred_df[self.cols]


    def plot_one_chr(self, sampled, pred_type, r1=0, r2=10):
        subset = sampled[r1:r2]
        if pred_type == "lnr_ext":
            pred_ = self.predict_single_linear_extend(subset)
        if pred_type == "lnr_mid":
            pred_ = self.predict_single_linear_middle(subset)

        ax = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.plot(*subset[self.coors].values.T, "-y")

        for _, row in subset.iterrows():
            ax.text(*row[self.coors].values, int(row[self.m]), color="y")

        ax.plot(*pred_[self.coors].values.T, ".r")

        for _, row in pred_.iterrows():
            ax.text(*row[self.coors].values, int(row[self.m]), color="r")


    def generate_imputed_others(self, out_path, pred_type):
        self.avg_MSE(pred_type, 1.0)

        final = pd.concat([self.data] + self.predicted[pred_type])
        final = final.drop_duplicates(self.cols[:2]).sort_values(self.cols[:2])
        final.to_csv(out_path, index=False, sep="\t")
    
    
    def generate_imputed_lnr_ext(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        pref = re.sub(r"^.*(129|CAST|mESC.*chr\d+).*$", "\g<1>", self.coor_path)

        i = 0
        while i == 0 or len(prev) > len(self.data):
            self.data, i = prev if i != 0 else self.data, i + 1
            self.avg_MSE("lnr_ext", 1.0)
            final = pd.concat([self.data] + self.predicted["lnr_ext"])
            prev = final.drop_duplicates(self.cols[:2]).sort_values(self.cols[:2])
            iter_path = os.path.join(out_path, f"{pref}_lnr_ext_iter{i}_pred.txt")
            prev.to_csv(iter_path, index=False, sep="\t")

        self.preprocessing()


    def generate_imputed_data(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        method_path = os.path.join(out_path, "lnr_mid.txt")
        self.generate_imputed_others(method_path, "lnr_mid")
        method_path = os.path.join(out_path, "spl_raw.txt")
        self.generate_imputed_others(method_path, "spl_raw")

        # method_path = os.path.join(out_path, "spl_opt")
        # self.generate_imputed_others(method_path, "spl_opt")
        method_path = os.path.join(out_path, "lnr_ext")
        self.generate_imputed_lnr_ext(method_path)


    def apply_summ(self, arr):
        return [np.nanmean(arr), np.nanmedian(arr), 
                np.nanstd(arr), np.nanmin(arr), np.nanmax(arr)]


    def measure_distance_deviation(self):
        # repeat 5 times for randomly drawn samples
        for i in range(5):
            for pred_type in self.mde_list:
                self.avg_MSE(pred_type, 0.9)

        lnr_ext = self.mde_list["lnr_ext"]
        lnr_ext = np.array([t + (2102-len(t))*[np.nan] for t in lnr_ext]).T
        lnr_mid = np.array(self.mde_list["lnr_mid"]).T
        spl_raw = np.array(self.mde_list["spl_raw"]).T
        spl_opt = np.array(self.mde_list["spl_opt"]).T

        final = np.hstack([lnr_ext, lnr_mid, spl_raw, spl_opt])
        names = ["lnr_ext", "lnr_mid", "spl_raw", "spl_opt"]
        col_names = [t + f"_{i}" for t in names for i in range(1, 6)]
        final = pd.DataFrame(final, columns=col_names)
        final.to_csv("output/1119_dist_measure.txt", sep="\t", index=False)

        out = list(map(self.apply_summ, [lnr_ext, lnr_mid, spl_raw, spl_opt]))
        out = pd.DataFrame(out, columns=["mean", "median", "std", "min", "max"],
                        index=names, dtype=float)
        out.round(3).to_csv("output/1119_mde_summary.txt", sep="\t")