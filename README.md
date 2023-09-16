# SnapFISH-IMPUTE: an imputation method for multiplexed DNA FISH data

### Usage

To run SnapFISH-IMPUTE on a computing cluster, include the following command in your bash script:
```{python}
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
* `SUF`: file with suffix