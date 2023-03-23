import os, argparse, re
from itertools import combinations
import pandas as pd
import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize

class ClassIterator:
    def __init__(self, class_num, tmp_dire, ann_dire):
        self.c, self.h, self.coor = "cell_id", "pos", ["x", "y", "z"]
        self.ann = pd.read_csv(ann_dire, sep="\t").rename({
            "Region ID":self.h, "hyb":self.h, "pos":self.h, 
            "Start":"start", "End":"end"
        }, axis=1)
        self.class_num, self.tmp_dire = class_num, tmp_dire
        self.preprocess()

    def preprocess(self):
        nimp_dire = os.path.join(self.tmp_dire, f"n{self.class_num}.txt")
        self.nimp_data = pd.read_csv(nimp_dire, sep="\t")
        data_dire = os.path.join(self.tmp_dire, f"d{self.class_num}.txt")
        self.data = pd.read_csv(data_dire, sep="\t")
        self.pos_range = np.arange(min(self.ann[self.h]), max(self.ann[self.h])+1)

        info_df = pd.read_csv(
            os.path.join(self.tmp_dire, "info.txt"), index_col=0, sep="\t"
        ).reset_index(drop=True)
        self.iter_diff_thresh = info_df.loc[0, "iter_diff_thresh"]
        self.data_min = pd.Series({
            "x":info_df.loc[0, "min_x"], 
            "y":info_df.loc[0, "min_y"], 
            "z":info_df.loc[0, "min_z"]
        }, dtype="float64")
        self.data_range = info_df.loc[0, "range"]

        self.nimp_mean_df = pd.read_csv(os.path.join(self.tmp_dire, "mean_df.txt"), sep="\t")

    def apply_dist(self, x):
        diff = np.array([*combinations(x[self.coor].values, 2)])
        return np.sqrt(np.sum(np.square(diff[:,0,:] - diff[:,1,:]), axis=1))
    
    def mean_df(self, data):
        return self.nimp_mean_df

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
            prev_data, l, sub_data = sub_data.copy(), 0, sub_data.copy()
            print(f"cell {k}:", end="\t")
            while l == 0 or iter_diff > self.iter_diff_thresh:
                for i in miss_pos:
                    sub_data.loc[i, self.coor] = minimize(
                        fun=self.score, args=(sub_data, mean_df, i),
                        x0=sub_data.loc[i][self.coor].values
                    ).x
                iter_diff = np.sum(np.abs(sub_data[self.coor].values - \
                    prev_data[self.coor].values))
                print(round(iter_diff * 1e3, 3), end="\t")
                prev_data, l = sub_data.copy(), l+1
            print()
            updated_clu_df.append(sub_data.reset_index())
        return pd.concat(updated_clu_df)
    
    def generate_update_one_class(self):
        updated_df = self.update_one_class(self.data)
        updated_df[self.coor] = updated_df[self.coor] * self.data_range + self.data_min
        out_path = os.path.join(self.tmp_dire, f"e{self.class_num}.txt")
        updated_df.to_csv(out_path, sep="\t", index=False)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--chr", action="store", required=True, help="chromosome number")
    parser.add_argument("-i", "--indir", action="store", required=True, help="input directory")
    parser.add_argument("-o", "--outdir", action="store", required=True, help="output directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_parser()
    lnr_dire = os.path.join(args.outdir, "lnr_imputed")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_class = max([int(re.sub(r"^\D*(\d+).*$", "\g<1>", f)) 
        for f in os.listdir("parallel_tmp"+args.chr) if f.startswith("d")]) + 1
    for k in np.arange(rank, n_class, comm.Get_size()):
        ann_dire = os.path.join(args.indir, f"ann_chr{args.chr}.txt")
        ci = ClassIterator(k, "parallel_tmp"+args.chr, ann_dire)
        ci.generate_update_one_class()