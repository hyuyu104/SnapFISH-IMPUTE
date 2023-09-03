from mpi4py import MPI
import argparse
from sfimpute.parallel import parallel_wrapper


def creat_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdire", action = "store", help = "output directory")
    parser.add_argument("-d", "--coor", action = "store", help = "3D coordinates path")
    parser.add_argument("-a", "--ann", action = "store", help = "1D annotation file path")
    parser.add_argument("-s", "--suf", action = "store", help = "suffix of file names")
    return parser


def main():
    args = creat_parser().parse_args()
    if args.outdire == "split":
        for i in range(1, 10):
            parallel_wrapper(
                MPI=MPI, output_dire=f"output_{args.suf}", 
                suf=f"{args.suf}_{i}",
                coor_wnan_path=f"data_mBCC_seqFISH_1Mb/mBCC_seqFISH_1Mb_{i}_coor_wnan.txt", 
                ann_path=args.ann,
            )
    else:
        parallel_wrapper(
            MPI=MPI, output_dire=f"output_{args.suf}", suf=args.suf,
            coor_wnan_path=args.coor, ann_path=args.ann,
        )


if __name__ == "__main__":
    main()