#!/bin/sh
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=shantanughosh@ufl.edu   # Where to send mail
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=32gb                   # Memory limit
#SBATCH --time=20:05:00               # Time limit hrs:min:sec

#SBATCH --output=Regressor_log_%j.out   # Standard output and error log

#SBATCH --account=cis6930
#SBATCH --qos=cis6930
#SBATCH --partition=gpu
#SBATCH --gpus=1

pwd; hostname; date

module load python

echo "Deep Colorizer"

python3 -u main.py > Colorizer_Kalpak_Shantanu.out

date
