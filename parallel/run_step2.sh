#!/bin/bash
#SBATCH --job-name=129_mr_impute_full_iter
#SBATCH -o /proj/yunligrp/users/hongyuyu/AggImpute/logfiles/0_%x_%j.out
#SBATCH -N 2
#SBATCH -p debug_queue
#SBATCH --ntasks-per-node=44
#SBATCH -t 4:00:00
#SBATCH --mem=200g
#SBATCH --mail-type END
#SBATCH --mail-user hongyuyu@unc.edu

input_dire="0913GenerateData/data_Sox2_seqFISH"
out_dire="0913GenerateData/0202_Sox2_mragg"
chr=1
method="iterate"
suffix="itern"
pythonpath="/proj/yunligrp/users/hongyuyu/LociImputation/venv/bin/"

python3 ClassGenerator.py -n $chr -i $input_dire -o $out_dire -k generate
module load openmpi_3.1.4/intel_18.2
unset PYTHONPATH
export PATH=/nas/longleaf/home/hongyuyu/.conda/envs/snaphic/bin/:$PATH

mpiexec -n 88 python3 ClassIterator.py -n $chr -i $input_dire -o $out_dire -m $method

unset PYTHONPATH
export PATH=$pythonpath:$PATH
python3 ClassGenerator.py -n $chr -i $input_dire -o $out_dire -k combine -s $suffix