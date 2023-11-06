# SnapFISH-IMPUTE: an imputation method for multiplexed DNA FISH data

SnapFISH-IMPUTE fills in the missing 3D coordinates of imaging loci in multiplexed DNA FISH data.

## Installation

SnapFISH-IMPUTE is available on PyPI and can be installed through `pip`. To create a virtual environment before installation, use either `conda`:
```bash
conda create --name sfimpute_env python==3.9.1
conda activate sfimpute_env
pip install sfimpute
```
or if on a computing cluster:
```bash
module load python/3.9.1
python -m venv /PATH/TO/ENV
source /PATH/TO/ENV/bin/activate
pip install sfimpute
```
Although `MPI` and `mpi4py` are needed to run the imputation module, the package on `PyPI` is independent of `MPI`, so you can still call functions in SnapFISH-IMPUTE even if MPI is not available.

To install `mpi4py`, please follow the instructions in [this link](https://mpi4py.readthedocs.io/en/stable/install.html).

## Usage

### Imputation

To run the imputation module on a computing cluster, download `run_impute.py` and include the following command when submitting the job:
```bash
mpiexec -np 50 python run_impute.py -o $OUT/DIRE -d $COORPATH -a $ANNPATH -s $SUF
```
where
* `OUT/DIRE`: the directory to store imputation results
* `COORPATH`: the path of the 3D coordinates file (.txt file separated by `\t`) with the following columns
    * region: imaging region ID
    * haploid: haploid ID (unique within each imaging region)
    * pos: locus ID (starts from 1)
    * x, y, z: 3D coordinates in nm (missing values are replaced by NaN)
* `ANNPATH`: the path of the annotation file (.txt file separated by `\t`) with the following columns
    * region: imaging region ID
    * pos: locus ID (starts from 1)
    * start: starting 1D genomic location of the imaging locus
    * end: ending 1D genomic location of the imaging locus
* `SUF`: file suffix

The jupyter notebook [`preprcess.ipynb`](jupyter/preprocess.ipynb) shows how to convert the imaging data to the desired form.

The output will be
```bash
$ tree OUT/DIRE
OUT/DIRE/
├── linear_coor_SUF.txt   # linear imputation output
└── recover_coor_SUF.txt  # SnapFISH-IMPUTE result
```

### Normalization of Pairwise Distances

SnapFISH-IMPUTE includes a normalization module to remove 1D genomic distance bias from the data and transform the distribution to approximately `N(0,1)`. This module can be called in Python by
```python
data = sfimpute.preprocess.read_data("PATH/TO/COOR.txt")
ann = sfimpute.preprocess.read_data("PATH/TO/ann.txt")

pdist_df = sfimpute.impute.to_dist_df(data, ann)
norm_df = sfimpute.impute.normalize_pdist_by1d(pdist_df)
```
For a more detailed tutorial about how pairwise distances are calculated and stored, please check [`pdistdemo.ipynb`](jupyter/pdistdemo.ipynb).

## Example

Use the 5kb chromatin tracing data of mESCs ([Huang et al. 2021](https://www.nature.com/articles/s41588-021-00863-6)) as an example. The formatted data is in the folder [`data`](data). Using 50 processes:
```bash
mpiexec -np 50 python run_impute.py -o output 
                                    -d data/mESC_Sox2_coor_wnan.txt 
                                    -a data/mESC_Sox2_ann.txt 
                                    -s mESCs_5kb
```
The program will print how many values are still unavailable in each iteration:
```
Region 129 initial: 590610 NaN values
Region 129 resized by a factor of 2
Region 129: 56640 NaN values
Region 129 resized by a factor of 2
Region 129: 56640 NaN values
Region CAST initial: 547912 NaN values
Region CAST resized by a factor of 2
Region CAST: 0 NaN values
```
After the program finished, a directory named output with the following files will be generated
```bash
$ tree output
output/
├── linear_coor_mESCs_5kb.txt   # linear imputation output
└── recover_coor_mESCs_5kb.txt  # SnapFISH-IMPUTE result
```

## Contact Us

If you encounter any problem while running SnapFISH-IMPUTE or have any questions regarding the rationale, you can send emails to Hongyu Yu (hongyuyu@unc.edu).
