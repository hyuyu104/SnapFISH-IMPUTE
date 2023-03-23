# SnapFISH-IMPUTE: an imputation algorithm for multiplexed DNA FISH data

## Introduction

SnapFISH-IMPUTE is an algorithm that can fill in all missing data in a given multiplexed DNA FISH dataset. It requires two inputs: an imaging dataset containg 3D coordinates of loci and a 1D genomic annotation file for imaged loci.

## Dependencies

SnapFISH-IMPUTE was developed with the following packages:

- numpy==1.21.5
- pandas==1.4.2
- matplotlib==3.5.1
- scipy==1.7.3
- scikit-learn==1.0.2

## Running SnapFISH-IMPUTE

Please define the following vairables in `run_impute.sh` before running SnapFISH-IMPUTE:

1. `nm_coor_path`: the file path of the input 3D coordinates. The file should have the following columns:

    - `cell_id`: unique ID for each cell
    - `pos`: unique ID for each locus
    - `x`, `y`, `z`: columns containing the 3D coordinates in nm of each locus in each cell

2. `nm_ann_path`: the file path of 1D genomic annotations. The file should contain:

    - `chr`: the chromosome ID of each locus
    - `start`: the starting genomic location of each locus
    - `end`: the ending genomic location of each locus
    - `pos`: unique ID for each locus, same as in `1`.

3. `step`: either `1` or `2`. If `1`, will generate the linear imputation result and the SnapFISH-IMPUTE step `1` result. If `2`, will generate the SnapFISH-IMPUTE step `2` result. Step `2` can only be run after step `1` since it dependes on the output of step `1`.

4. `lnr_path`: the name of the output file generated by linear imputation.

5. `step1_path`: the name of the output file generated by the 1st step of SnapFISH-IMPUTE.

6. `step2_path`: the name of the output file generated by the 2nd step of SnapFISH-IMPUTE. This is the final output of SnapFISH-IMPUTE. It contains the 3D coordinates of all loci and has the same format as the input. 

7. `py_path`: python3 path

8. `wrapper_path`: the path of `impute_wrapper.py`

After defining the required variables, run the shell script using `sh run_impute.sh`. For a complete imputation process, run the script twice, with the first time setting `step=1` and the second time setting `step=2`.

For large datasets, a more efficient implementation using MPI for python is provided in `parallel`. It requires running the script on a computing cluster.

## Files

Multiplexed DNA FISH data from Huang et al ([PMID: 34002095](https://pubmed.ncbi.nlm.nih.gov/34002095/), 5Kb resolution) and Takei et al ([PMID: 33505024](https://pubmed.ncbi.nlm.nih.gov/33505024/), 25Kb resolution) are included in `data/mESC_sox2_raw` and `data/mESC_seqFISH+_raw`, respectively. These two datasets are then processed using `IMPUTE/preprocess.py` to rename columns and convert to nm. The processed files are stored in `data/mESC_sox2_nm` and `data/mESC_seqFISH+_nm`.

Imputation results of the 5Kb resolution dataset are in `output/mESC_sox2`, where suffices `lnr`, `lnr_ext`, `spl`, `step1`, and `step2` correspond to linear imputation, linear extension, smooth spline, the 1st step of SnapFISH-IMPUTE, and the 2nd step of SnapFISH-IMPUTE.

Imputation results of the 25Kb resolution dataset are in `output/mESC_seqFISH+`. Chromosomes are processed separately and then combined together, resulting in the three files listed in the folder.