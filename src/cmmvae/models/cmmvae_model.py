from typing import Optional

import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer  # type: ignore

from cmmvae.models import BaseModel
from cmmvae.modules import CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules.base.components import GradientReversalFunction
from cmmvae.config import AutogradConfig

from cmmvae.data.local.species_dataset import NPZDataset


class CMMVAEModel(BaseModel):
    """
    Conditional Multi-Modal Variational Autoencoder (CMMVAE) model for handling expert-specific data.

    This class is designed for training VAEs with multiple experts and adversarial components.

    Args:
        module (Any): Conditional Multi-Modal VAE module.
        batch_size (int, optional): Batch size for logging purposes only. Defaults to 128.
        record_gradients (bool, optional): Whether to record gradients of the model. Defaults to False.
        save_gradients_interval (int): Interval of steps to save gradients. Defaults to 25.
        gradient_record_cap (int, optional): Cap on the number of gradients to record to prevent clogging TensorBoard. Defaults to 20.
        kl_annealing_fn (KLAnnealingFn, optional): Annealing function used for kl_weight. Defaults to `KLAnnealingFn(1.0)`
        predict_dir (str): Directory to save predictions. If not absolute path then saved within Tensorboard log_dir. Defaults to "".
        predict_save_interval (int): Interval to save embeddings and metadata to prevent OOM Error. Defaults to 600.
        initial_save_index (int): The starting point for predictions index when saving (ie z_embeddings_0.npz for -1). Defaults to -1.
        use_he_init_weights (bool): Initialize weights using He initialization. Defaults to True.

    Attributes:
        module (`CMMVAE`): The CMMVAE module for processing and generating data.
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
        adversarial_criterion (nn.CrossEntropyLoss): Loss function for adversarial training.
        kl_annealing_fn (cmmvae.modules.base.KLAnnealingFn): KLAnnealingFn for weighting KL Divergence. Defaults to KLAnnealingFn(1.0).
    """

    def __init__(
        self,
        module: CMMVAE,
        adv_weight=None,
        adversarial_method="",
        autograd_config: Optional[AutogradConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module = module
        self.automatic_optimization = (
            False  # Disable automatic optimization for manual control
        )
        # Criterion for adversarial loss
        if adversarial_method == "GRF":
            adv_criterion = cross_entropy_one_hot_labels
        else:
            adv_criterion = nn.BCELoss(reduction="sum")

        self.adversarial_criterion = adv_criterion

        self.init_weights()
        self.adv_weight = adv_weight if adv_weight else 1.0
        self.adversarial_method = adversarial_method
        self.autograd_config = autograd_config or AutogradConfig()



    def _liam_grf(
            self,
            adversarial_group,
            hidden_representations: list[torch.Tensor],
            label: torch.Tensor,
            loss_dict: dict,
    ):
        """
        Calculate the loss for an adversarial group 
        """

        assert (
            len(hidden_representations)
            == len(adversarial_group.adversarials)
        )
        label = label.to(self.device)
        losses = []

        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, adversarial_group.adversarials)
        ):

            adv = adv.to(self.device)

            if adversarial_group.conditional != "cell_type":
                hidden_rep = GradientReversalFunction.apply(hidden_rep, 1)

            adv_output = adv(hidden_rep)

            back_prop_loss = self.adversarial_criterion(adv_output, label, reduction="mean")
            mean_loss = self.adversarial_criterion(adv_output, label, reduction="mean")

            losses.append(back_prop_loss)

            loss_dict[RK.ADV_LOSS + adversarial_group.conditional + str(i)] = mean_loss
        
        return torch.stack(losses).sum()


    def liam_grf(
        self,
        hidden_representations: list[torch.Tensor],
        labels,
        main_loss_dict: dict,
    ):
        assert self.module.adversarial_groups
        adv_group_losses = []
        for adv_group in self.module.adversarial_groups:
            loss = self._liam_grf(adv_group, hidden_representations,
                                              labels[adv_group.conditional], main_loss_dict)
            adv_group_losses.append(loss)

        return torch.stack(adv_group_losses).sum()

    def get_same_cell_type_centroids(self, same_cell_types_dict: dict, expert_id: str) -> None:
        """
        Get embeddings for all cell types in the batch and store them in the same_cell_types_dict.

        Args:
            same_cell_types_dict (dict): Dictionary mapping cell types to their corresponding embeddings.
            expert_id (str): The ID of the expert for which to get the embeddings.
        """
        centroid_dict = {}
        for cell_type in same_cell_types_dict.keys():
            raw_data: torch.Tensor = same_cell_types_dict[cell_type]
            self.module.eval()
            embeddings = self.module.get_latent_embeddings(
                raw_data.to(self.device),
                pd.DataFrame({"cell_type": [cell_type]}),
                expert_id
            )
            self.module.train()
            centroid = embeddings[RK.Z][0].mean(dim=0, keepdim=True)
            centroid_dict[cell_type] = centroid

        return centroid_dict


    def training_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str, list[torch.Tensor]], batch_idx: int
    ) -> None:
        x, metadata, expert_id, labels = batch

        # Get optimizers
        optims = self.get_optimizers()
        expert_optimizer = optims["experts"][expert_id]
        vae_optimizer = optims["vae"]

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Perform forward pass
        qz, pz, z, xhats, hidden_representations = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        # Calculate reconstruction loss
        main_loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )
        total_loss = main_loss_dict[RK.LOSS]

        adv_loss = None
        if self.module.adversarial_groups and self.current_epoch >= 0:
            adv_loss = self.liam_grf(
                hidden_representations,
                labels,
                main_loss_dict,
            )

        if adv_loss is not None:
            main_loss_dict[RK.ADV_LOSS] = adv_loss
            total_loss += adv_loss * self.adv_weight

        snnl_loss = snnl(hidden_representations[0], labels["cell_type"])

        main_loss_dict["SNNL"] = snnl_loss

        total_loss += snnl_loss * 1000
        self.manual_backward(total_loss, retain_graph=True)

        self.log_gradient_norms(
            {"vae": vae_optimizer, f"expert_{expert_id}": expert_optimizer},
            tag_prefix="grad_norms/main_network/total_gradients",
        )

        if self.module.adversarial_groups:
            for adv_group in self.module.adversarial_groups:
                for i, adv in enumerate(adv_group.adversarials):
                    self.log_gradient_norms(
                        {f"adversarial_{adv_group.conditional}_{i}": self.get_optimizers()["adversarials" + adv_group.conditional][i]},
                        tag_prefix="grad_norms/adversarial_networks",
                    )

        if self.autograd_config.vae_gradient_clip:
            self.clip_gradients(vae_optimizer, *self.autograd_config.vae_gradient_clip)

        if self.autograd_config.expert_gradient_clip:
            self.clip_gradients(
                expert_optimizer, *self.autograd_config.expert_gradient_clip
            )


        vae_optimizer.step()
        expert_optimizer.step()
        if self.module.adversarial_groups:
            self.step_adv_optimizers()

        vae_optimizer.zero_grad()
        expert_optimizer.zero_grad()
        if self.module.adversarial_groups:
            for adv_group in self.module.adversarial_groups:
                self.zero_adv_optimizers(adv_group.conditional)


        self.kl_annealing_fn.step()

        # Log the loss
        self.auto_log(main_loss_dict, tags=[self.stage_name, expert_id])

    def validation_step(self, batch: tuple[torch.Tensor, pd.DataFrame, str]):
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id, _ = batch
        # expert_label = self.module.experts.labels[expert_id]

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, hidden_representations = self.module(x, metadata, expert_id)

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

        if self.trainer.validating:
            self.log("val_loss", loss_dict[RK.LOSS], logger=False, on_epoch=True)

    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def predict_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ):
        """
        Perform a prediction step.

        This step extracts latent embeddings and saves them for analysis.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        x, metadata, species, _ = batch
        embeddings = self.module.get_latent_embeddings(x, metadata, species)
        return embeddings
        # self.save_predictions(embeddings, batch_idx)

    def get_optimizers(self, zero_all: bool = False):
        """
        Retrieve optimizers for the model components.

        This function resets gradients if specified and returns a structured dictionary of optimizers.

        Args:
            zero_all (bool, optional): Flag to reset gradients of all optimizers. Defaults to False.

        Returns:
            dict: Dictionary containing optimizers for experts, VAE, and adversarials.
        """
        optimizers = self.optimizers()

        if zero_all:
            for optim in optimizers:  # type: ignore
                optim.zero_grad()

        def replace_indices_with_optimizers(mapping, optimizer_list):
            if isinstance(mapping, dict):
                return {
                    key: replace_indices_with_optimizers(value, optimizer_list)
                    for key, value in mapping.items()
                }
            else:
                return optimizer_list[mapping]

        # Create a dictionary with indices replaced with optimizer instances
        optimizer_dict = replace_indices_with_optimizers(self.optimizer_map, optimizers)

        return optimizer_dict

    def configure_optimizers(self, optim_cls="Adam") -> list[Optimizer]:  # type: ignore
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optim_cls = Adam if optim_cls == "Adam" else AdamW
        optim_dict = {}
        optim_dict["experts"] = {
            expert_id: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
            for expert_id, module in self.module.experts.items()
        }
        optim_dict["vae"] = optim_cls(
            self.module.vae.parameters(), lr=5e-3, weight_decay=1e-6
        )

        if self.module.adversarial_groups:
            for adversarial_group in self.module.adversarial_groups:
                optim_dict["adversarials" + adversarial_group.conditional ] = {
                    i: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
                    for i, module in enumerate(adversarial_group.adversarials)
                }

        optimizers = []
        self.optimizer_map = convert_to_flat_list_and_map(optim_dict, optimizers)

        return optimizers

    def step_adv_optimizers(self):
        optimizers = self.get_optimizers()
        for group in self.module.adversarial_groups:
            for _, opt in optimizers["adversarials" + group.conditional].items():
                if self.autograd_config.adversarial_gradient_clip:
                    self.clip_gradients(opt, *self.autograd_config.adversarial_gradient_clip)
                opt.step()

    def zero_adv_optimizers(self, condition):
        optimizers = self.get_optimizers()
        for _, opt in optimizers["adversarials" + condition].items():
            opt.zero_grad()

def convert_to_flat_list_and_map(d: dict, flat_list: Optional[list] = None) -> dict:
    """
    Convert all values in the dictionary to a flat list and return the list and a mapping dictionary.
    Args:
        d (dict): The dictionary to convert.
        flat_list (list, optional): The list to append values to. Defaults to None.

    Returns:
        dict: Mapping dictionary linking keys to indices in the flat list.
    """
    if flat_list is None:
        flat_list = []

    map_dict = {}

    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            map_dict[key] = convert_to_flat_list_and_map(value, flat_list)
        else:
            # Add value to flat list and set its index in the mapping
            flat_list.append(value)
            map_dict[key] = len(flat_list) - 1

    return map_dict

def cross_entropy_one_hot_labels(predictions, labels, reduction="sum"):
    integer_labels = torch.argmax(labels, dim=1)
    return torch.nn.functional.cross_entropy(predictions, integer_labels, reduction=reduction)
    
def pairwise_cos_distance(A, B):
    query_embeddings = torch.nn.functional.normalize(A, dim=1)
    key_embeddings = torch.nn.functional.normalize(B, dim=1)
    distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
    return distances

def snnl(embeddings, labels, temperature=1.0):
    """
    Args:
        embeddings: Batched embeddings to compute the SNNL.
        labels: Labels of embeddings.
    """

    #convert one hot labels to integer labels
    labels = torch.argmax(labels, dim=1)

    batch_size = embeddings.shape[0]
    eps = 1e-9
    
    pairwise_dist = pairwise_cos_distance(embeddings, embeddings)
    pairwise_dist = pairwise_dist / temperature
    negexpd = torch.exp(-pairwise_dist)

    # creating mask to sample same class neighboorhood
    pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
    mask = pairs_y == torch.transpose(pairs_y, 0, 1)
    mask = mask.float()

    # creating mask to exclude diagonal elements
    ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
    dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

    # all class neighborhood
    alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
    # same class neighborhood
    sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

    # adding eps for numerical stability
    # in case of a class having a single occurrence in batch
    # the quantity inside log would have been 0
    loss = -torch.log((sacn+eps)/alcn).mean()
    return loss


