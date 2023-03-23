
nm_coor_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/data/mESC_sox2_nm/mESC_129_allele.txt"
nm_ann_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/data/mESC_sox2_nm/input_ann.txt"

step=2
lnr_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/output/mESC_sox2/mESC_sox2_a129_lnr.txt"
step1_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/output/mESC_sox2/mESC_sox2_a129_step1.txt"
step2_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/output/mESC_sox2/mESC_sox2_a129_step2.txt"

py_path="/Users/hongyudemacpro/opt/anaconda3/bin/python"
wrapper_path="/Users/hongyudemacpro/Desktop/uw/lab/biostat/SnapFISH-IMPUTE_UPLOAD/impute_wrapper.py"

# $py_path IMPUTE/preprocess.py

$py_path $wrapper_path -c $nm_coor_path -a $nm_ann_path -s $step -o1 $step1_path -o2 $step2_path -l $lnr_path