import h5py

key = "z"
# with h5py.File("/mnt/projects/debruinz_project/cardell_taylor/test_backprop/MMVAE_Adversarial_Summer/lightning_logs/zebrafish/conditionals.test.local20250611_234329/predictions.h5", "r") as f:
with h5py.File("/mnt/projects/debruinz_project/cardell_taylor/test_backprop/MMVAE_Adversarial_Summer/lightning_logs/zebrafish/conditionals.test.local20250611_231536/predictions.h5", "r") as f:
    print(f"Groups under {key}: {list(f[key].keys())}")
    
    for ds_name in f[key]:
        ds = f[key][ds_name]
        if isinstance(ds, h5py.Dataset):
            print(f"{ds_name}: shape = {ds.shape}")
        elif isinstance(ds, h5py.Group):
            print(f"{ds_name} is a group with keys: {list(ds.keys())}")