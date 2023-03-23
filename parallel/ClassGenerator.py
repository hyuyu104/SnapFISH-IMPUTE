import os, argparse
import pandas as pd
from AggCluImpute import AggCluImpute

class ClassGenerator(AggCluImpute):
    def __init__(self, coor_dire, nimp_dire, ann_dire, tmp_dire):
        self.tmp_dire = tmp_dire
        if not os.path.exists(tmp_dire):
            os.mkdir(tmp_dire)
        super().__init__(coor_dire, nimp_dire, ann_dire, "pos")

    def create_clusters(self):
        super().create_clusters()
        for class_num, data_df in self.data.groupby("label"):
            data_df.to_csv(
                os.path.join(self.tmp_dire, f"d{class_num}.txt"),
                sep = "\t", index = False
            )
            self.nimp_data[self.nimp_data[self.c].isin(data_df[self.c])].to_csv(
                os.path.join(self.tmp_dire, f"n{class_num}.txt"),
                sep = "\t", index = False
            )
        pd.DataFrame({
            "iter_diff_thresh":[self.iter_diff_thresh],
            "min_x":[self.data_min["x"]],
            "min_y":[self.data_min["y"]],
            "min_z":[self.data_min["z"]],
            "range":[self.data_range]
        }).to_csv(os.path.join(self.tmp_dire, "info.txt"), sep="\t")

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--chr", action="store", required=True, help="chromosome number")
    parser.add_argument("-i", "--indir", action="store", required=True, help="input directory")
    parser.add_argument("-o", "--outdir", action="store", required=True, help="output directory")
    parser.add_argument("-k", "--step", action="store", required=True, help="step name")
    parser.add_argument("-s", "--suffix", action="store", required=False, help="file suffix")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_parser()
    # lnr_dire = os.path.join(args.outdir, "lnr_imputed")
    lnr_dire = os.path.join(args.outdir, "mr_imputed")
    if args.step == "generate":
        cg = ClassGenerator(
            # coor_dire = os.path.join(lnr_dire, f"mESC_lnr_mid_chr{args.chr}_pred.txt"),
            coor_dire = os.path.join(lnr_dire, f"mESC_mr_update_chr{args.chr}_pred.txt"),
            nimp_dire = os.path.join(args.indir, f"mESC_seqFISH_chr{args.chr}.txt"),
            ann_dire = os.path.join(args.indir, f"ann_chr{args.chr}.txt"),
            tmp_dire = "parallel_tmp"+args.chr
        )
        cg.create_clusters()
    elif args.step == "combine":
        combined_df = pd.concat([
            pd.read_csv(os.path.join("parallel_tmp"+args.chr, f), sep="\t") 
            for f in os.listdir("parallel_tmp"+args.chr) if f.startswith("e")
        ]).reset_index(drop=True).sort_values(["cell_id", "pos"])
        out_path = os.path.join(args.outdir, "mESC_agg_"+args.suffix+"_chr"+args.chr+"_pred.txt")
        combined_df.to_csv(out_path, sep="\t", index=False)
        os.system("rm -r parallel_tmp"+args.chr)