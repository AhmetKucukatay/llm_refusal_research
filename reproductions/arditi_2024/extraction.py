"""Activation extraction for the Arditi et al. reproduction."""

from collections.abc import Callable, Sequence

import torch
from torch import nn


def extract_resid_pre_activations( #resid_pre = residual stream activations before the transformer block's attention and feedforward layers
    model: nn.Module,
    block_modules: Sequence[nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    """Extract block-input activations at selected token positions.

    Args:
        model: Causal language model used for the forward pass.
        block_modules: Transformer blocks in forward order.
        input_ids: Token IDs with shape ``(batch_size, sequence_length)``.
        attention_mask: Attention mask with the same first two dimensions.
        positions: Token positions to extract, such as ``[-5, -4, ..., -1]``.

    Returns:
        Residual-stream activations with shape
        ``(batch_size, num_positions, num_layers, hidden_dim)``.
    """
    if len(block_modules) == 0:
        raise ValueError("At least one transformer block must be provided.")

    if len(positions) == 0:
        raise ValueError("At least one token position must be provided.")

    #One slot per transformer layer; each pre-hook fills its layer's resid_pre.
    layer_activations: list[torch.Tensor | None] = [
        None
        for _ in block_modules
    ]

    handles = []

    def make_hook(layer: int): #we eill hook each layer seperately, then make_hook(0) => hook for layer 0
        def hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            del module

            resid_pre = inputs[0]  #get block first input tensor before entering block => resid_pre

            #resid_pre shape = (batch_size, sequence_length, hidden_dim), with below indexing we get only -5, -4 ... tokenpositions
            #Keep only the post-instruction token positions used to form candidates.
            layer_activations[layer] = resid_pre[:, positions, :].detach()

        return hook  #make hook returns a function, not a hook itself, we are doing like this because block 0 -> hook for layer 0..

    try:
        for layer, block in enumerate(block_modules):
            handle = block.register_forward_pre_hook(make_hook(layer))
            handles.append(handle)

        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
    finally:
        for handle in handles:
            handle.remove()

    if any(activation is None for activation in layer_activations):
        raise RuntimeError("Failed to capture activations from every block.")

    return torch.stack(
        [
            activation
            for activation in layer_activations
            if activation is not None
        ],
        dim=2,
    )
# N instructions-> batch batch tokenzie-> extract_resid_pre_activation-> for qwen each batch [B,5,24,2048] -> sum on sample dimension-> global mean -> [5,24,2048]
def compute_mean_resid_pre_activations(
    model: nn.Module,
    block_modules: Sequence[nn.Module],
    instructions: Sequence[str],
    tokenize_instructions_fn: Callable,
    positions: Sequence[int],
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute mean resid_pre activations over a set of instructions."""
    if len(instructions) == 0:
        raise ValueError("At least one instruction must be provided.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    n_samples = len(instructions)

    mean_activations = None

    for start in range(0, n_samples, batch_size):
        batch_instructions = instructions[start : start + batch_size] #create batches
        inputs = tokenize_instructions_fn(batch_instructions)  #tokenize batches

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        batch_activations = extract_resid_pre_activations(
            model=model,
            block_modules=block_modules,
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
        )
        batch_sum = batch_activations.to(torch.float64).sum(dim=0)

        #first batch extraction-> sum(dim=0) -> [P,L,D] -> this weill be mean tensors shape. 
        #this part is neccesary because now function does not depend on model.confix-hidden_size which is for HF
        if mean_activations is None:
            mean_activations = torch.zeros_like(batch_sum)
        # Accumulate in float64, matching the released implementation's
        # high-precision mean activation cache.
        mean_activations += (
            batch_activations.to(torch.float64).sum(dim=0) / n_samples
        )

    if mean_activations is None:
        raise RuntimeError("Failed to compute mean activations.")
    return mean_activations