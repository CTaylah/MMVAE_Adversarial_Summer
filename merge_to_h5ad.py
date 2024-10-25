import numpy as np
import pickle
import anndata as ad
import pandas as pd

# Paths to the files
# npz_file = "/active/debruinz_project/cardell_taylor/MMVAE/lightning_logs/experiment/version000.full.local20241006_230148/merged/z_embeddings.npz"
# metadata_file = "/active/debruinz_project/cardell_taylor/MMVAE/lightning_logs/experiment/version000.full.local20241006_230148/merged/z_metadata.pkl"

log_path = "/active/debruinz_project/cardell_taylor/MMVAE/lightning_logs/experiment/balancing_adv_weight_0.95_kl_weight_5.0.full.local20241012_130019/"

npz_file = log_path + "merged/z_embeddings.npz"
metadata_file = log_path + "/merged/z_metadata.pkl"
# Step 1: Load single-cell data from .npz file
npz_data = np.load(npz_file, allow_pickle=True)
npz_data = npz_data['embeddings']

# Step 2: Define a custom persistent_load function to handle persistent IDs
def persistent_load(pid):
    # Handle the persistent ID references (e.g., you can map it to a default object or handle it more specifically)
    print(f"Encountered persistent ID: {pid}")
    return None  # Handle external references appropriately, modify as needed

# Step 3: Use the Unpickler with a custom persistent_load to load the metadata
with open(metadata_file, 'rb') as f:
    unpickler = pickle.Unpickler(f)
    unpickler.persistent_load = persistent_load
    metadata = unpickler.load()

print(npz_data)
# Step 4: Create an AnnData object with the loaded data
adata = ad.AnnData(X=npz_data)

# Step 5: Set metadata (obs) in the AnnData object
adata.obs = pd.DataFrame(metadata)

# Step 6: Save the combined data as a .h5ad file
adata.write('adv_weight_0.95_kl_weight_5.0.h5ad')
