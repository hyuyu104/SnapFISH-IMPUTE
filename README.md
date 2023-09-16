# SnapFISH-IMPUTE: an imputation method for multiplexed DNA FISH data

### Usage

To run SnapFISH-IMPUTE on a computing cluster, include the following command in your bash script:
```bash
mpiexec -np 50 python run_impute.py -o $OUT/DIRE -d $COORPATH -a $ANNPATH -s $SUF
```
where
* `OUT/DIRE`: the directory to store imputation results
* `COORPATH`: the path of the 3D coordinates file (.txt file separated by `\t`) with the following columns
    * chr: imaging region ID
    * cell_id: haploid chromosome ID (unique within each imaging region)
    * pos: locus ID (starts from 1)
    * x, y, z: 3D coordinates in nm (missing values are replaced by NaN)
* `ANNPATH`: the path of the annotation file (.txt file separated by `\t`) with the following columns
    * chr: imaging region ID
    * pos: locus ID (starts from 1)
    * start: starting 1D genomic location of the imaging locus
    * end: ending 1D genomic location of the imaging locus
* `SUF`: file suffix

Use the 5kb chromatin tracing data of mESCs ([Huang et al. 2021](https://www.nature.com/articles/s41588-021-00863-6)) as an example:
```bash
mpiexec -np 50 python run_impute.py -o output 
                                    -d data/mESC_Sox2_coor_wnan.txt 
                                    -a data/mESC_Sox2_ann.txt 
                                    -s mESCs_5kb
```
A directory named output with the following files will be generated
```bash
$ tree output
output/
├── linear_coor_mESCs_5kb.txt           # linear imputation output
├── recover_coor_mESCs_5kb.txt          # SnapFISH-IMPUTE result
├── target_dist_mESCs_5kb_reg129.txt    # target pairwise distances for the 129 allele
└── target_dist_mESCs_5kb_regcast.txt   # target pairwise distances for the CAST allele
```