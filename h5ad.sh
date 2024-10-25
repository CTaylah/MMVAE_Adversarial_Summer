#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=179G ##Memory I want to use in MB

#SBATCH --time=24:00:00 ## time it will take to complete job

#SBATCH --partition=all ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=hm ## Name of job

#SBATCH --cpus-per-task=40 

source /active/debruinz_project/cardell_taylor/MMVAE/env/bin/activate
python3 merge_to_h5ad.py
