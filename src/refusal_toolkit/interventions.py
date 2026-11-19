"""Generic activation intervention utilities for refusal research."""
from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
from torch import nn


def ablate_direction(
    activation: torch.Tensor,
    direction: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Remove the component of an activation along a direction."""
    unit_direction = direction / (direction.norm() + epsilon)
    unit_direction = unit_direction.to(activation) # ensure unit_direction is on the same device and dtype as activation

    projection = (activation @ unit_direction).unsqueeze(-1) * unit_direction
    return activation - projection


def add_direction(
    activation: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Add a direction vector to every token activation."""
    direction = direction.to(activation)
    return activation + coefficient * direction


def make_direction_ablation_pre_hook(
    direction: torch.Tensor,
) -> Callable:
    """Create a pre-hook that removes a direction from a module input."""

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ):
        activation = inputs[0]
        ablated = ablate_direction(activation, direction)

        return (ablated, *inputs[1:])

    return hook


def make_direction_ablation_output_hook( #function that returns a hook function that remembers direction
    direction: torch.Tensor,
) -> Callable:
    """Create a hook that removes a direction from a module output."""

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output,
    ):
        if isinstance(output, tuple):
            activation = output[0]
            ablated = ablate_direction(activation, direction)
            return (ablated, *output[1:])

        return ablate_direction(output, direction) #remove direction from attention output

    return hook


def make_activation_addition_pre_hook( #this is prehook, before transformer block add the direction
    direction: torch.Tensor,
    coefficient: float = 1.0,
) -> Callable:
    """Create a pre-hook that adds a direction to a module input."""

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ):
        activation = inputs[0]
        modified = add_direction(
            activation,
            direction,
            coefficient=coefficient,
        )

        return (modified, *inputs[1:])

    return hook

#so far we can create hooks and attach this to a module(if prehook, manipulate activation before transformer blocks, if output after MLP and Attn)
# , but registered hook will not disappear. What we want is for candidate 1, append hook, forward, remove hook then continue with candidate 2
@contextmanager
def temporary_hooks(
    pre_hooks: list[tuple[nn.Module, Callable]],
    output_hooks: list[tuple[nn.Module, Callable]],
) -> Generator[None, None, None]:
    """Temporarily register hooks and remove them after use."""
    handles = []

    try:
        for module, hook in pre_hooks:
            handles.append(
                module.register_forward_pre_hook(hook)
            )

        for module, hook in output_hooks:
            handles.append(
                module.register_forward_hook(hook)
            )

        yield

    finally:  #make sure that we remove the hooks
        for handle in handles:
            handle.remove()