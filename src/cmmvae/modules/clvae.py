from typing import Optional

import torch
from torch.distributions import Normal, kl_divergence, Distribution
import pandas as pd

from cmmvae.modules.vae import VAE
from cmmvae.modules.base import FCBlockConfig, ConditionalLayers, ConcatBlockConfig, TiedConditionalLayers


class CLVAE(VAE):
    """
    Conditional Latent Variational Autoencoder class.

    This class extends the basic VAE to incorporate conditional layers,
    allowing for conditioning the latent space on additional metadata.

    Args:
        encoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the decoder's fully connected block.
        conditional_config (Optional[cmmvae.modules.base.FCBlockConfig]):
            Configuration for the conditional layers.
        conditional_paths (Dict[str, str]):
            Mapping of conditional paths for the conditional layers.
        selection_order (Optional[List[str]]):
            Order in which to apply selection of conditionals.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(
        self,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
        conditional_config: Optional[FCBlockConfig] = None,
        conditionals_directory: Optional[str] = None,
        conditionals: Optional[list[str]] = None,
        selection_order: Optional[list[str]] = None,
        concat_config: Optional[ConcatBlockConfig] = None,
        **encoder_kwargs
    ):
        conditionals_module = None
        if conditional_config and conditionals and conditionals_directory:
            conditionals_module = ConditionalLayers(
                directory=conditionals_directory,
                conditionals=conditionals,
                fc_block_config=conditional_config,
                selection_order=selection_order,
            )
        else:
            import warnings

            warnings.warn("No conditionals found for vae")

        if selection_order and selection_order[0] == "parallel":
            if not concat_config:
                raise RuntimeError(
                    "Please define concat_config when selection_order = parallel"
                )
            concat_dim = (
                len(conditionals_module.selection_order) * conditional_config.layers[-1]
            )

            decoder_config.layers = [concat_dim] + decoder_config.layers
            decoder_config.activation_fn = [
                concat_config.activation_fn
            ] + decoder_config.activation_fn
            decoder_config.dropout_rate = [
                concat_config.dropout_rate
            ] + decoder_config.dropout_rate
            decoder_config.return_hidden = [
                concat_config.return_hidden
            ] + decoder_config.return_hidden
            decoder_config.use_layer_norm = [
                concat_config.use_layer_norm
            ] + decoder_config.use_layer_norm
            decoder_config.use_batch_norm = [
                concat_config.use_batch_norm
            ] + decoder_config.use_batch_norm

        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **encoder_kwargs,
        )

        self.conditionals = conditionals_module
        self.tied_conditionals = TiedConditionalLayers(self.conditionals)

    def after_reparameterize(
        self, z: torch.Tensor, metadata: pd.DataFrame, **kwargs
    ) -> torch.Tensor:
        """
        Modify the latent variable after reparameterization
            by applying conditional layers.

        If conditional layers are defined, they will be applied to
            the latent variable `z` using the provided `metadata`.

        Args:
            z (torch.Tensor): Latent variable of shape (batch_size, n_latent).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            torch.Tensor:
                Processed latent variable after applying conditionals, if any.
        """
        residual = z
        beta = 0.1
        if self.conditionals:
            conditional_output = self.conditionals(
                z, metadata, **kwargs
            )
            conditional_output += residual * beta

            return self.tied_conditionals(
                conditional_output, metadata, **kwargs
            )
        # Return the unmodified latent variable
        # if no conditionals are present
        return z

    # def encode(self, x: torch.Tensor, metadata: pd.DataFrame, **kwargs):
    #     """
    #     Encode the input tensor into a latent representation.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, n_in).

    #     Returns:
    #         tuple:
    #             - qz (Distribution): The approximate posterior distribution
    #                 over the latent space.
    #             - z (torch.Tensor): Sampled latent variable from qz.
    #             - hidden_representations (List[torch.Tensor]):
    #                 List of hidden representations from the encoder.
    #     """

    #     encoder_block_out = self.encoder.encode(x)
    #     if isinstance(encoder_block_out, tuple):
    #         encoder_block_out, _ = encoder_block_out

    #     z = self.conditionals(encoder_block_out, metadata, **kwargs) if self.conditionals else encoder_block_out
    #     return super().encode(z, **kwargs)

    # def forward(self, x: torch.Tensor, metadata: pd.DataFrame, **kwargs):

    #     encoder_block_out = self.encoder.encode(x)
    #     hidden_representations = []
    #     if isinstance(encoder_block_out,tuple):
    #         encoder_block_out, hidden_representations = encoder_block_out

    #     residual = encoder_block_out
    #     x = (self.conditionals(encoder_block_out, metadata, **kwargs) + residual) if self.conditionals else encoder_block_out
    #     qz, z, _ = super().encode(x, **kwargs)
    #     hidden_representations.append(z)

    #     pz = Normal(torch.zeros_like(z), torch.ones_like(z))
    #     z_mod = self.after_reparameterize(z, metadata, **kwargs)
    #     xhat = self.decode(z_mod, **kwargs)
    #     return qz, pz, z_mod, xhat, hidden_representations