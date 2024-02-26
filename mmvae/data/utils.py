import scipy.sparse as sp
import pandas as pd
import torch

_CELL_CENSUS_COLUMN_NAMES = ["soma_joinid","dataset_id","assay","cell_type","development_stage","disease","donor_id","self_reported_ethnicity","sex","tissue","tissue_general"]

def split_data_and_metadata(data_file_path: str, metadata_file_path: pd.DataFrame, train_ratio: float):
    """
    Splits a csr_matrix and its corresponding metadata (pandas DataFrame) into training and validation sets based on a given ratio.

    :param data_file_path: The file path in which to load .npz.
    :param metadata_file_path: The file path to the pandas DataFrame containing metadata corresponding to the csr_matrix's rows.
    :param train_ratio: A float between 0 and 1 indicating the ratio of data to be used for training.
    :return: A tuple containing the training and validation sets for both the csr_matrix and the DataFrame.
    """
    matrix: sp.csr_matrix = sp.load_npz(data_file_path)
    metadata = pd.read_csv(metadata_file_path, header=None, names=_CELL_CENSUS_COLUMN_NAMES)
    # Ensure the train_ratio is within the valid range and the lengths match
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if matrix.shape[0] != len(metadata):
        raise ValueError("The number of rows in the matrix and metadata must match")

    # Calculate the split index
    split_index = int(matrix.shape[0] * train_ratio)

    # Split the matrix
    train_data = matrix[:split_index]
    train_data = torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape)
    validation_data = matrix[split_index:]
    validation_data = torch.sparse_csr_tensor(validation_data.indptr, validation_data.indices, validation_data.data, validation_data.shape)

    # Split the metadata DataFrame
    train_metadata = metadata.iloc[:split_index]
    validation_metadata = metadata.iloc[split_index:]

    return (train_data, train_metadata), (validation_data, validation_metadata)