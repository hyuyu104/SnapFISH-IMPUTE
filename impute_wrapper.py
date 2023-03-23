import argparse
from IMPUTE.NaiveImputer import NaiveImputer
from IMPUTE.SF_IMPUTE import SF_IMPUTE_Step1, SF_IMPUTE_Step2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--in_coors', action = 'store', required = True)
    parser.add_argument('-a', '--in_anns', action = 'store', required = True)
    parser.add_argument('-s', '--step', action = 'store', required = True)
    parser.add_argument('-o1', '--step1_path', action = 'store', required = True)
    parser.add_argument('-o2', '--step2_path', action = 'store')
    parser.add_argument('-l', '--lnr_path', action = 'store')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.step == "1":
        ni = NaiveImputer(args.in_anns, args.in_coors)
        ni.generate_imputed_others(args.lnr_path, "lnr_mid")
        sis1 = SF_IMPUTE_Step1(args.lnr_path, args.in_coors, args.in_anns)
        sis1.create_clusters()
        sis1.generate_imputed(args.step1_path, False)
    if args.step == "2":
        sis2 = SF_IMPUTE_Step2(args.step1_path, args.in_coors, args.in_anns)
        sis2.create_clusters()
        sis2.generate_imputed(args.step2_path, True)