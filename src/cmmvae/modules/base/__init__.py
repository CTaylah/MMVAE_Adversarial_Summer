"""
    This module holds the building block nn.Modules and functions for training.
"""
from cmmvae.modules.base.components import (
    Encoder,
    FCBlock,
    FCBlockConfig,
    Expert,
    # TiedExpert,
    Experts,
    AdversarialsGroup,
    ConditionalLayer,
    TiedConditionalLayer,
    ConditionalLayers,
    TiedConditionalLayers,
    GradientReversalFunction,
    ConcatBlockConfig,
)

from cmmvae.modules.base.annealing_fn import KLAnnealingFn, LinearKLAnnealingFn

__all__ = [
    "AdversarialsGroup",
    "ConditionalLayer",
    "TiedConditionalLayer",
    "ConditionalLayers",
    "TiedConditionalLayers",
    "ConcatBlockConfig",
    "Encoder",
    "Expert",
    "Experts",
    "FCBlock",
    "FCBlockConfig",
    "GradientReversalFunction",
    "KLAnnealingFn",
    "LinearKLAnnealingFn",
]
