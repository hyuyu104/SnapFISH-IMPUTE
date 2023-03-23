#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=44
#SBATCH --mem=200g

input_dire="data_Sox2_seqFISH"
out_dire="0202_Sox2_mragg"
chr=1
method="iterate"
suffix="itern"
pythonpath="users/hongyuyu/LociImputation/venv/bin/"

python3 ClassGenerator.py -n $chr -i $input_dire -o $out_dire -k generate
module load openmpi_3.1.4/intel_18.2
unset PYTHONPATH
export PATH=/nas/longleaf/home/hongyuyu/.conda/envs/snaphic/bin/:$PATH

mpiexec -n 88 python3 ClassIterator.py -n $chr -i $input_dire -o $out_dire -m $method

unset PYTHONPATH
export PATH=$pythonpath:$PATH
python3 ClassGenerator.py -n $chr -i $input_dire -o $out_dire -k combine -s $suffix