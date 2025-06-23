#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --time=24:00:00 ## time it will take to complete job

#SBATCH --partition=cpu ##Partition I want to use

#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=82G

#SBATCH --job-name=conversion ## Name of job

source /mnt/projects/debruinz_project/cardell_taylor/test_backprop/MMVAE_Adversarial_Summer/env/bin/activate
python explore_h5.py