#!/bin/bash
#SBATCH --job-name=seqFISH_mr_chr19
#SBATCH -o /proj/yunligrp/users/hongyuyu/AggImpute/logfiles/0_%x_%j.out
#SBATCH -N 2
#SBATCH -p 528_queue
#SBATCH --ntasks-per-node=44
#SBATCH -t 2:00:00
#SBATCH --mem=200g
#SBATCH --mail-type END
#SBATCH --mail-user hongyuyu@unc.edu

input_dire="0913GenerateData/data_mESC_seqFISH_nm"
out_dire="0913GenerateData/0122_mESC_seqFISH_impute_nm"
suffix="update"
nproc=88
chr=19
pythonpath="/proj/yunligrp/users/hongyuyu/LociImputation/venv/bin/"

python3 MRClassGenerator.py -n $chr -i $input_dire -o $out_dire -k generate -p $nproc
module load openmpi_3.1.4/intel_18.2
unset PYTHONPATH
export PATH=/nas/longleaf/home/hongyuyu/.conda/envs/snaphic/bin/:$PATH

mpiexec -n $nproc python3 MRClassIterator.py -n $chr -i $input_dire -o $out_dire

unset PYTHONPATH
export PATH=$pythonpath:$PATH
python3 MRClassGenerator.py -n $chr -i $input_dire -o $out_dire -k combine -s $suffix -p $nproc