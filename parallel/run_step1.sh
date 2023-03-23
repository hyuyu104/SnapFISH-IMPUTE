#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=44
#SBATCH --mem=200g

input_dire="data_mESC_seqFISH_nm"
out_dire="0122_mESC_seqFISH_impute_nm"
suffix="update"
nproc=88
chr=19
pythonpath="users/hongyuyu/LociImputation/venv/bin/"

python3 MRClassGenerator.py -n $chr -i $input_dire -o $out_dire -k generate -p $nproc
module load openmpi_3.1.4/intel_18.2
unset PYTHONPATH
export PATH=/nas/longleaf/home/hongyuyu/.conda/envs/snaphic/bin/:$PATH

mpiexec -n $nproc python3 MRClassIterator.py -n $chr -i $input_dire -o $out_dire

unset PYTHONPATH
export PATH=$pythonpath:$PATH
python3 MRClassGenerator.py -n $chr -i $input_dire -o $out_dire -k combine -s $suffix -p $nproc