from typing import Union

from cmmvae.data.local.cellxgene_datapipe import SpeciesDataPipe
from cmmvae.data.encoding_dicts import assay_dict
from cmmvae.data.encoding_dicts import donor_id_dict
from cmmvae.data.encoding_dicts import dataset_id_dict
from cmmvae.data.encoding_dicts import cell_type_dict
import torch




def wrap_in_list(value):
    if isinstance(value, list):
        return value
    return [value]


class SpeciesManager:
    """
    SpeciesManager is responsible for managing data pipelines for species data.
    It initializes the configuration for training, validation, and testing datasets
    and provides methods to create data pipes with specific configurations.

    Attributes:
        directory_path (str): Path to the directory containing the datasets.
        train_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for training masks.
        train_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for training masks.
        val_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for validation masks.
        val_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for validation masks.
        test_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for test masks.
        test_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for test masks.
        batch_size (int): Batch size for data loading. Default is 128.
        return_dense (bool): Flag to return dense data. Default is False.
        verbose (bool): Flag for verbose output. Default is False.
        name (str): Name of the species manager instance.
    :no-index:
    """

    def __init__(
        self,
        name: str,
        directory_path: str,
        train_npz_masks: Union[str, list[str]],
        train_metadata_masks: Union[str, list[str]],
        val_npz_masks: Union[str, list[str]],
        val_metadata_masks: Union[str, list[str]],
        test_npz_masks: Union[str, list[str]],
        test_metadata_masks: Union[str, list[str]],
        batch_size: int = 128,
        return_dense: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the SpeciesManager with dataset paths and configurations.

        Args:
            name (str): Name of the species manager instance.
            directory_path (str): Path to the directory containing the datasets.
            train_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for training masks.
            train_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for training masks.
            val_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for validation masks.
            val_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for validation masks.
            test_npz_masks (Union[str, list[str]]): NPZ file paths or patterns for test masks.
            test_metadata_masks (Union[str, list[str]]): Metadata file paths or patterns for test masks.
            batch_size (int, optional): Batch size for data loading. Defaults to 128.
            return_dense (bool, optional): Flag to return dense data. Defaults to False.
            verbose (bool, optional): Flag for verbose output. Defaults to False.
        """
        super().__init__()
        self.directory_path = directory_path
        self.train_npz_masks = wrap_in_list(train_npz_masks)
        self.val_npz_masks = wrap_in_list(val_npz_masks)
        self.train_metadata_masks = wrap_in_list(train_metadata_masks)
        self.val_metadata_masks = wrap_in_list(val_metadata_masks)
        self.test_npz_masks = wrap_in_list(test_npz_masks)
        self.test_metadata_masks = wrap_in_list(test_metadata_masks)
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.verbose = verbose
        self.name = name

    def transform_fn(self):
        """
        Creates a transformation function for the data pipeline.

        Returns:
            function: A generator function that processes source data and appends the species name.
        """

        species_mapping= {
            "human": 0,
            "mouse": 1,
        }

        def encode_conditional(conditional: str, mapping: dict, species=False) -> torch.tensor:
            if species:
                if self.name == "human":
                    encoded_labels = torch.zeros(self.batch_size, dtype=torch.long)
                elif self.name == "mouse":
                    encoded_labels = torch.ones(self.batch_size, dtype=torch.long)
                else:
                    raise ValueError("Unexpected condition given")
            else:
                encoded_labels = torch.tensor([mapping[elt] for elt in conditional], dtype=torch.long)

            # One-hot encode the tensor
            one_hot_matrix = torch.nn.functional.one_hot(encoded_labels, num_classes=len(mapping))

            return one_hot_matrix

        def generator(source):
            tensor, metadata = source

            assay_values = metadata["assay"].values
            donor_id_values = metadata["donor_id"].values
            dataset_id_values = metadata["dataset_id"].values

            one_hot_donor_id = encode_conditional(donor_id_values, donor_id_dict.donor_id)
            one_hot_dataset_id = encode_conditional(dataset_id_values, dataset_id_dict.dataset_id)
            one_hot_assay = encode_conditional(assay_values, assay_dict.assay)
            one_hot_cell_type = encode_conditional(metadata["cell_type"].values, cell_type_dict.cell_type)
            one_hot_species = encode_conditional([self.name], species_mapping, species=True)
            
            one_hot_labels = {
                "donor_id": one_hot_donor_id,
                "dataset_id": one_hot_dataset_id,
                "assay": one_hot_assay,
                "cell_type": one_hot_cell_type,
                "species": one_hot_species
            }



            return tensor, metadata, self.name, one_hot_labels

        return generator


    def train_datapipe(self):
        """
        Creates a data pipeline for training data.

        Returns:
            SpeciesDataPipe: A data pipe configured for training data with shuffling enabled.
        """
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.train_npz_masks,
            metadata_masks=self.train_metadata_masks,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn(),
        )

    def val_datapipe(self):
        """
        Creates a data pipeline for validation data.

        Returns:
            SpeciesDataPipe: A data pipe configured for validation data with shuffling disabled.
        """
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.val_npz_masks,
            metadata_masks=self.val_metadata_masks,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn(),
        )

    def test_datapipe(self):
        """
        Creates a data pipeline for testing data.

        Returns:
            SpeciesDataPipe: A data pipe configured for testing data with shuffling disabled.
        """
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.test_npz_masks,
            metadata_masks=self.test_metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn(),
        )

    def predict_datapipe(self):
        """
        Creates a data pipeline for prediction data.

        Returns:
            SpeciesDataPipe: A data pipe configured for prediction data with shuffling disabled.
        """
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.test_npz_masks,
            metadata_masks=self.test_metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn(),
        )
