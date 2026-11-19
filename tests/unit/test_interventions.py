"""Unit tests for activation intervention utilities."""
import torch
from torch import nn

from refusal_toolkit.interventions import (
    ablate_direction,
    add_direction,
    make_activation_addition_pre_hook,
    make_direction_ablation_output_hook,
    make_direction_ablation_pre_hook,
    temporary_hooks,
)


def test_ablate_direction_removes_parallel_component() -> None:
    """The ablation pre-hook modifies a module input."""
    activation = torch.tensor([[[3.0, 4.0]]])
    direction = torch.tensor([2.0, 0.0])

    result = ablate_direction(activation, direction)

    expected = torch.tensor([[[0.0, 4.0]]])
    assert torch.allclose(result, expected)


def test_add_direction() -> None:
    """The activation addition pre-hook modifies a module input."""
    activation = torch.tensor([[[3.0, 4.0]]])
    direction = torch.tensor([1.0, 2.0])

    result = add_direction(
        activation,
        direction,
        coefficient=2.0,
    )

    expected = torch.tensor([[[5.0, 8.0]]])
    assert torch.allclose(result, expected)


def test_ablation_pre_hook_modifies_module_input() -> None:
    """The ablation pre-hook modifies a module input."""
    module = nn.Identity()

    direction = torch.tensor([1.0, 0.0])
    hook = make_direction_ablation_pre_hook(direction)

    handle = module.register_forward_pre_hook(hook)

    activation = torch.tensor([[[3.0, 4.0]]])
    result = module(activation)

    handle.remove()

    expected = torch.tensor([[[0.0, 4.0]]])
    assert torch.allclose(result, expected)


def test_ablation_output_hook_modifies_module_output() -> None:
    """The ablation output hook modifies a module output."""
    module = nn.Identity()

    direction = torch.tensor([1.0, 0.0])
    hook = make_direction_ablation_output_hook(direction)

    handle = module.register_forward_hook(hook)

    activation = torch.tensor([[[3.0, 4.0]]])
    result = module(activation)

    handle.remove()

    expected = torch.tensor([[[0.0, 4.0]]])
    assert torch.allclose(result, expected)


def test_addition_pre_hook_modifies_module_input() -> None:
    """The addition pre-hook modifies a module input."""
    module = nn.Identity()

    direction = torch.tensor([1.0, 2.0])
    hook = make_activation_addition_pre_hook(
        direction,
        coefficient=2.0,
    )

    handle = module.register_forward_pre_hook(hook)

    activation = torch.tensor([[[3.0, 4.0]]])
    result = module(activation)

    handle.remove()

    expected = torch.tensor([[[5.0, 8.0]]])
    assert torch.allclose(result, expected)

def test_temporary_hooks_are_removed_after_context() -> None:
    """Temporary hooks affect only forwards inside the context."""
    module = nn.Identity()

    direction = torch.tensor([1.0, 0.0])
    hook = make_direction_ablation_pre_hook(direction)

    activation = torch.tensor([[[3.0, 4.0]]])

    with temporary_hooks(
        pre_hooks=[(module, hook)],
        output_hooks=[],
    ):
        inside_result = module(activation)

    outside_result = module(activation)

    expected_inside = torch.tensor([[[0.0, 4.0]]])
    expected_outside = torch.tensor([[[3.0, 4.0]]])

    assert torch.allclose(inside_result, expected_inside)
    assert torch.allclose(outside_result, expected_outside)