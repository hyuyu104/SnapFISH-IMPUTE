from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy.spatial.distance as D
from sklearn.cluster import AgglomerativeClustering

class SF_IMPUTE_Step2:
    def __init__(self, coor_dire, nimp_dire, ann_dire, h="pos"):
        """
        Parameters
        ----------
        coor_dire : str
            Step 1 result file path
        nimp_dire : str
            Raw data (no imputation performed) file path
        ann_dire: str
            The path of the 1D genomic annotation file
        h: str, optional
            The name of the loci ID column in input files, default "pos"
        """
        self.c, self.h, self.coor = "cell_id", h, ["x", "y", "z"]

        rename_dict = {
            "Region ID":self.h, "hyb":self.h, "pos":self.h, 
            "Start":"start", "End":"end"
        }

        nimp_data = pd.read_csv(nimp_dire, sep="\t")
        self.nimp_data = nimp_data.rename(rename_dict, axis=1)

        data = pd.read_csv(coor_dire, sep="\t")
        self.data = data.rename(rename_dict, axis=1)

        ann = pd.read_csv(ann_dire, sep="\t")
        self.ann = ann.rename(rename_dict, axis=1)

        self.preprocess()
    

    def preprocess(self):
        ann_range = np.ptp(self.ann[self.h])
        assert ann_range == np.ptp(self.data[self.h])
        assert ann_range == np.ptp(self.nimp_data[self.h])

        # pos in ann, data, nimp_data should have same range
        min_pos = min(self.ann[self.h])
        adj_pos1 = min_pos - min(self.data[self.h])
        self.data[self.h] = self.data[self.h] + adj_pos1

        adj_pos2 = min_pos - min(self.nimp_data[self.h])
        self.nimp_data[self.h] = self.nimp_data[self.h] + adj_pos2

        self.pos_range = np.arange(min(self.ann[self.h]), max(self.ann[self.h])+1)

        self.miss_sr = self.nimp_data.groupby(self.c).apply(
            lambda x: np.ptp(self.ann[self.h]) + 1 - x.shape[0]
        )

        self.data_min = self.data[self.coor].min()
        self.data[self.coor] = self.data[self.coor] - self.data_min
        self.data_range = self.data[self.coor].mean(axis=1).median()
        self.data[self.coor] = self.data[self.coor]/self.data_range

        nimp_min = self.nimp_data[self.coor].min()
        self.nimp_data[self.coor] = self.nimp_data[self.coor] - nimp_min
        nimp_data_range = self.nimp_data[self.coor].mean(axis=1).median()
        self.nimp_data[self.coor] = self.nimp_data[self.coor]/nimp_data_range


    def apply_dist(self, x):
        diff = np.array([*combinations(x[self.coor].values, 2)])
        return np.sqrt(np.sum(np.square(diff[:,0,:] - diff[:,1,:]), axis=1))


    def create_clusters(self):
        self.dist_sr = self.data.groupby(self.c).apply(self.apply_dist)
        self.iter_diff_thresh = np.mean(self.dist_sr.tolist())*len(self.pos_range)/1e2

        ac = AgglomerativeClustering(
            n_clusters=int(len(np.unique(self.data[self.c]))/2),
            linkage="ward",
            # distance_threshold=5*(np.mean(self.dist_sr.tolist())**2)
        ).fit(self.dist_sr.tolist())
        self.labels_ = ac.labels_

        dist_idx_np = np.array(list(enumerate(self.dist_sr.index)))
        dist_dict = dict(np.concatenate(
            [dist_idx_np[:, [1]], dist_idx_np[:, [0]]], axis=1
        ))
        self.data["label"] = self.data.apply(
            lambda x: ac.labels_[dist_dict[x[self.c]]], axis=1
        )
    

    def visualize_one_class(self, label_n):
        nrows = (Counter(self.labels_)[label_n]-1)//4+1
        fig, ax = plt.subplots(ncols=5, nrows=nrows, figsize=(10, 2*nrows))
        ax = ax.flatten()
        list(map(lambda x: x.axis("off"), ax))

        dist_np = self.dist_sr[self.labels_ == label_n].mean()
        ax[0].imshow(-dist_matrix(dist_np, len(self.pos_range)), cmap="RdBu")
        ax = ax.reshape((-1, 5))

        for i in range(Counter(self.labels_)[label_n]):
            dist_np = self.dist_sr[self.labels_ == label_n].iloc[i]
            ax[i//4,1+i%4].imshow(-dist_matrix(dist_np, len(self.pos_range)), cmap="RdBu")
            ax[i//4,1+i%4].text(-2, -2, "num miss: " + \
                str(self.miss_sr[self.labels_ == label_n].iloc[i])
            )


    def mean_df(self, data):
        mean_dist = data.groupby(self.c).apply(self.apply_dist).mean()
        bin_idx = np.array(tuple(combinations(self.pos_range, 2)))
        return pd.DataFrame({"bin1":bin_idx[:, 0], "bin2":bin_idx[:, 1], "dist":mean_dist})


    def score(self, new_coor, sub_data, mean_df, i):
        coor_diff = sub_data[sub_data.index!=i][self.coor].values - new_coor
        diff_dist = np.sqrt(np.sum(np.square(coor_diff), axis=1))
        mean_dist_i = mean_df[(mean_df["bin1"]==i)|(mean_df["bin2"]==i)]["dist"]
        return sum(np.square(diff_dist - mean_dist_i))


    def update_one_class(self, data):
        mean_df = self.mean_df(data)
        updated_clu_df = []
        for k, sub_data in data.set_index(self.h).groupby(self.c):
        
            nimp_pos_range = self.nimp_data[self.nimp_data[self.c]==k][self.h]
            miss_pos = sorted(list(set(self.pos_range) - set(nimp_pos_range)))
            prev_data, l = sub_data.copy(), 0
            print(f"cell {k}:", end="\t")

            while l == 0 or iter_diff > self.iter_diff_thresh:

                for i in miss_pos:
                    sub_data.loc[i, self.coor] = minimize(
                        fun=self.score, args=(sub_data, mean_df, i),
                        x0=sub_data.loc[i][self.coor].values
                    ).x

                d = D.pdist(sub_data[self.coor].values) - mean_df["dist"]
                print(f"({round(np.linalg.norm(d)*1e3, 4)})", end="")

                iter_diff = np.sum(np.abs(sub_data[self.coor].values - \
                    prev_data[self.coor].values))
                print(round(iter_diff * 1e3, 3), end="\t")
                prev_data, l = sub_data.copy(), l+1

            print()
            updated_clu_df.append(sub_data.reset_index())
        return pd.concat(updated_clu_df)


    def iterate_one_class(self, label_n):
        print("*"*20, f"class {label_n}", "*"*20)
        
        class_df = self.data[self.data["label"]==label_n]
        class_size = len(np.unique(class_df[self.c]))
        prev_mean_df, k = self.mean_df(class_df), 0
        self.iter_clu_thresh = self.iter_diff_thresh * 4

        while (k == 0 or mean_diff > self.iter_clu_thresh) and k < 30:

            if k == 0:
                new_class_data = self.update_one_class(class_df)
            else:
                new_class_data = self.update_one_class(new_class_data)

            mean_df = self.mean_df(new_class_data)
            mean_diff = np.sum(np.abs(prev_mean_df["dist"] - mean_df["dist"]))
            print(f"cluster iter #{k}:", mean_diff * 1e3)
            prev_mean_df, k = mean_df, k + 1

        return new_class_data


    def visualize_one_class_updated(self, updated_df):
        sub_dist_df = updated_df.groupby(self.c).apply(self.apply_dist)

        n_cells = len(np.unique(updated_df[self.c]))
        nrows = (n_cells-1)//4+1
        fig, ax = plt.subplots(ncols=5, nrows=nrows, figsize=(10, 2*nrows))
        ax = ax.flatten()
        list(map(lambda x: x.axis("off"), ax))

        dist_mat = dist_matrix(sub_dist_df.mean(), len(self.pos_range))
        ax[0].imshow(-dist_mat, cmap="RdBu")
        ax = ax.reshape((-1, 5))

        for i in range(n_cells):
            dist_mat = dist_matrix(sub_dist_df.iloc[i], len(self.pos_range))
            ax[i//4,1+i%4].imshow(-dist_mat, cmap="RdBu")
            ax[i//4,1+i%4].text(-2, -2, "num miss: " + \
                str(self.miss_sr[sub_dist_df.index[i]])
            )
    

    def generate_imputed(self, out_path, w_loop):
        self.result = []
        for n in np.unique(self.labels_):
            if w_loop:
                res = self.iterate_one_class(n)
            else:
                res = self.update_one_class(self.data[self.data["label"]==n])
            res[self.coor] = res[self.coor] * self.data_range + self.data_min
            self.result.append(res)

        result_df = pd.concat(self.result).sort_values([self.c, self.h])
        result_df.to_csv(out_path, sep="\t", index=False)



class SF_IMPUTE_Step1(SF_IMPUTE_Step2):
    def preprocess(self):
        super().preprocess()
        self.nimp_mean_df = pd.DataFrame()


    def create_clusters(self, n_proc = None):
        self.dist_sr = self.data.groupby(self.c).apply(self.apply_dist)
        self.iter_diff_thresh = np.mean(self.dist_sr.tolist())*len(self.pos_range)/1e2
        self.labels_ = np.repeat(0, len(np.unique(self.data[self.c])))

        # cell_ids = np.unique(self.data["cell_id"])
        # class_dict = dict(zip(cell_ids, np.arange(len(cell_ids)) % n_proc))
        # self.data["label"] = self.data["cell_id"].apply(lambda x: class_dict[x])
        self.data["label"] = 0


    def mean_df(self, data):
        if len(self.nimp_mean_df) == 0:
            data = self.nimp_data

            CELL_IDS = np.unique(data[self.c])
            cell_ids = np.repeat(CELL_IDS, self.ann.shape[0])
            pos_ids = np.tile(self.ann[self.h], len(CELL_IDS))

            missed_rows = set(zip(cell_ids, pos_ids)) - \
                set(zip(data[self.c], data[self.h]))
            missed_rows = np.array(list(missed_rows))
            missed_df = pd.DataFrame(missed_rows, columns=[self.c, self.h])

            data = pd.concat([data, missed_df])
            data = data.sort_values([self.c, self.h]).reset_index(drop=True)

            dists = data.groupby(self.c).apply(self.apply_dist)
            mean_dist = np.nanmean(np.array(dists.values.tolist()), axis=0)
            bin_idx = np.array(tuple(combinations(self.pos_range, 2)))

            self.nimp_mean_df = pd.DataFrame(
                {"bin1":bin_idx[:, 0], "bin2":bin_idx[:, 1], "dist":mean_dist}
            )
        return self.nimp_mean_df
    

def dist_matrix(dist_np, n):
    bins = np.array([*combinations(np.arange(0, n), 2)])
    dist_df = np.concatenate([bins, dist_np.reshape((-1, 1))], axis=1)
    dist_df = pd.DataFrame(dist_df, columns = ["bin1", "bin2", "dist"])
    pivot = pd.pivot_table(dist_df, "dist", "bin1", "bin2").values
    dist_mat = np.diag([0.0]*n)
    dist_mat[np.triu_indices(n, 1)] = pivot[np.triu_indices_from(pivot)]
    dist_mat[np.tril_indices(n, -1)] = pivot.T[np.tril_indices_from(pivot)]
    return dist_mat