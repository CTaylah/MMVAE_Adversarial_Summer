import pandas as pd

mouse_dataframe = pd.read_csv("/mnt/projects/debruinz_project/cardell_taylor/scripts/mouse_marker_genes.csv")
human_dataframe = pd.read_csv("/mnt/projects/debruinz_project/cardell_taylor/scripts/3_files_marker_genes.csv")

mouse_marker_dict = mouse_dataframe.to_dict(orient='list')
human_marker_dict = human_dataframe.to_dict(orient='list')
