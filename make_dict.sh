#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=1024 ##Memory I want to use in MB

#SBATCH --time=00:07:00 ## time it will take to complete job

#SBATCH --partition=cpu ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=dictionary ## Name of job

#SBATCH --output=dictionary.%j.out ##Name of output file

module module load py-venv-ml/nightly 
python make_dict.py --file_path /mnt/projects/debruinz_project/july2024_census_data/subset/ --label cell_type

