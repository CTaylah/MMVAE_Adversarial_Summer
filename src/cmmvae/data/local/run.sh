#!/bin/bash
#SBATCH --partition=all
#SBATCH --mem=140G
#SBATCH --time=120:00:00
#SBATCH --job-name=scib

source /mnt/projects/debruinz_project/cardell_taylor/fall_pipeline/env/bin/activate
python3 species_dataset.py
