import torch
import pickle
from torch.utils.data import Dataset

import os
import time
import numpy as np
from scipy.sparse import csr_matrix

import pandas

class NPZDataset(Dataset):
    def __init__(self, directory, file_name, metadata_name, transform=None):
        """
        Args:
            directory (str): Path to the directory containing .npz files.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.directory = directory
        self.file_path = file_name
        self.metadata_path = metadata_name
        self.transform = transform

        self.metadata: pandas.DataFrame

        self.data = np.load(directory + "/" + self.file_path[0])
        indices = self.data['indices']
        indptr = self.data['indptr']
        values = self.data['data']
        shape = tuple(self.data['shape'])
        data_format = self.data['format'].item()
        if self.data['format'].item() == b'csr':
            self.data_matrix = torch.sparse_csr_tensor(
                torch.tensor(indptr, dtype=torch.int64),
                torch.tensor(indices, dtype=torch.int64),
                torch.tensor(values, dtype=torch.float32),
                size=shape)
            # self.data_matrix.cuda()
        else:
            raise ValueError("Only supports csr format")

        with open(directory + "/" + self.metadata_path[0], 'rb') as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.data_matrix.shape[0]

    def __getitem__(self, idx):
        return self.data_matrix[idx].to_dense(), self.metadata.iloc[idx]

    def get_same_cell_types(self, batch_metadata: pandas.DataFrame):

        # Start timer for the entire method
        start_time = time.time()
        print("Starting get_same_cell_types...")

        # Build a list of samples with the same cell type for each sample in the batch
        batch_cell_types = batch_metadata['cell_type']
        print(f"Batch cell types: {batch_cell_types}")

        # Create a dictionary to store matching cells for each cell type
        matching_cells_dict = {}

        for cell_type in batch_cell_types:
            cell_type_start_time = time.time()  # Timer for each cell type
            print(f"Processing cell type: {cell_type}")

            if cell_type not in matching_cells_dict:
                # Grab a list of cells with the same cell type
                matching_indices_start_time = time.time()
                matching_indices = self.metadata[self.metadata['cell_type'] == cell_type].index.to_numpy()
                print(f"Time to find matching indices for cell type '{cell_type}': {time.time() - matching_indices_start_time:.4f} seconds", flush=True)
                # print(f"Found {len(matching_indices)} matching indices for cell type '{cell_type}'", flush=True)

                # Timer for creating list_of_cells

                # list_start_time = time.time()
                # list_of_cells = [self.data_matrix[idx].to_dense() for idx in matching_indices]
                # print(f"Time to create list_of_cells for cell type '{cell_type}': {time.time() - list_start_time:.4f} seconds", flush=True)
                # print(f"Number of cells converted to dense for cell type '{cell_type}': {len(list_of_cells)}", flush=True)

                # Timer for creating matching_cells_data
                data_start_time = time.time()
                if len(matching_indices) == 0:
                    matching_cells_data = torch.zeros((1, self.data_matrix.shape[1]))
                    print(f"No matching cells found for cell type '{cell_type}', creating zero tensor")
                else:
                    # matching_cells_data = torch.stack(list_of_cells)
                    matching_cells_data = extract_rows_from_csr(self.data_matrix, matching_indices)
                    print(f"Stacked tensor shape for cell type '{cell_type}': {matching_cells_data.shape}", flush=True)
                print(f"Time to create matching_cells_data for cell type '{cell_type}': {time.time() - data_start_time:.4f} seconds", flush=True)

                matching_cells_dict[cell_type] = matching_cells_data

            print(f"Finished processing cell type '{cell_type}' in {time.time() - cell_type_start_time:.4f} seconds", flush=True)

        # End timer for the entire method
        print(f"Finished get_same_cell_types in {time.time() - start_time:.4f} seconds", flush=True)
        return matching_cells_dict


def extract_rows_from_csr(csr_matrix, row_indices):
    """
    Extracts the specified rows from a sparse CSR matrix and returns a dense tensor.

    Args:
        csr_matrix (torch.sparse_csr_tensor): The input sparse CSR matrix.
        row_indices (list or torch.Tensor): A list or tensor of row indices to extract.

    Returns:
        torch.Tensor: A dense tensor containing the extracted rows.
    """
    # Get the row and column indices and values of the CSR matrix
    crow_indices = csr_matrix.crow_indices()
    col_indices = csr_matrix.col_indices()
    values = csr_matrix.values()

    # Initialize an empty list to store the extracted rows
    extracted_rows = []

    # Loop over the row indices
    for idx in row_indices:
        # Get the range of indices for the current row
        start = crow_indices[idx].item()
        end = crow_indices[idx + 1].item()

        # Get the column indices and values for this row
        row_cols = col_indices[start:end]
        row_vals = values[start:end]

        # Create a sparse row tensor for the current row
        sparse_row = torch.sparse_csr_tensor(
            torch.tensor([0, len(row_vals)]),
            row_cols,
            row_vals,
            size=(1, csr_matrix.size(1))
        )
        
        # Convert the sparse row to dense format and append it to the list
        extracted_rows.append(sparse_row.to_dense())

    # Stack the extracted rows into a new dense matrix
    dense_matrix = torch.stack(extracted_rows)
    
    return dense_matrix