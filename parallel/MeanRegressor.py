from itertools import combinations
import pandas as pd
import numpy as np
from AggCluImpute import AggCluImpute

class MeanRegressor(AggCluImpute):

    def preprocess(self):
        super().preprocess()
        self.nimp_mean_df = pd.DataFrame()

    def create_clusters(self, n_proc):
        self.dist_sr = self.data.groupby(self.c).apply(self.apply_dist)
        self.iter_diff_thresh = np.mean(self.dist_sr.tolist())*len(self.pos_range)/1e2
        self.labels_ = np.repeat(0, len(np.unique(self.data[self.c])))

        cell_ids = np.unique(self.data["cell_id"])
        class_dict = dict(zip(cell_ids, np.arange(len(cell_ids)) % n_proc))
        self.data["label"] = self.data["cell_id"].apply(lambda x: class_dict[x])

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