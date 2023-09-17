from mpi4py import MPI
import argparse
from sfimpute.parallel import parallel_wrapper


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdire", action = "store", help = "output directory")
    parser.add_argument("-d", "--coor", action = "store", help = "3D coordinates path")
    parser.add_argument("-a", "--ann", action = "store", help = "1D annotation file path")
    parser.add_argument("-s", "--suf", action = "store", help = "suffix of file names")
    return parser


def main():
    args = create_parser().parse_args()
    parallel_wrapper(
        MPI=MPI, 
        output_dire=args.outdire, 
        suf=args.suf,
        coor_wnan_path=args.coor, 
        ann_path=args.ann,
    )


if __name__ == "__main__":
    main()