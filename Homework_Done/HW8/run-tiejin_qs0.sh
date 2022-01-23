#!/usr/bin/bash
#
# Author: Tiejin Chen
# Updated: Dec 03, 2021
# 1: -------------------------------------------------------------------------

# slurm options: --------------------------------------------------------------
#SBATCH --job-name=tiejin_ps8qs0
#SBATCH --mail-user=tiejin@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=10:00
#SBATCH --account=stats507f21_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/%x-%j-5.log

# application: ----------------------------------------------------------------

# modules 
#SBATCH --get-user-env

# the contents of this script
cat run-tiejin_qs0.sh

# run the script
date

cd /home/tiejin/
python PS8_chentiejin_qs0.py

date
echo "Done."