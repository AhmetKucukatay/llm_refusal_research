"""Tests for activation-selection utilities."""

from types import SimpleNamespace

import pytest
import torch

from refusal_toolkit.activations import (
    extract_hidden_states,
    gather_multiple_token_activations,
    gather_token_activations,
    get_last_non_padding_token_indices,
)


def test_gather_token_activations_returns_expected_vectors() -> None:
    """Selected token activations should match the expected batch-wise vectors."""
    activations = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    token_indices = torch.tensor([2, 0])

    selected = gather_token_activations(activations, token_indices)

    expected = torch.tensor(
        [
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    torch.testing.assert_close(selected, expected)


def test_gather_token_activations_rejects_non_3d_activations() -> None:
    """Activations must include batch, sequence, and hidden dimensions."""
    activations = torch.zeros((3, 4))
    token_indices = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match="three-dimensional"):
        gather_token_activations(activations, token_indices)


def test_gather_token_activations_rejects_non_1d_token_indices() -> None:
    """Token indices must contain exactly one position per batch item."""
    activations = torch.zeros((2, 3, 4))
    token_indices = torch.tensor([[0], [1]])

    with pytest.raises(ValueError, match="one-dimensional"):
        gather_token_activations(activations, token_indices)


def test_gather_token_activations_rejects_batch_size_mismatch() -> None:
    """The number of token indices must match the activation batch size."""
    activations = torch.zeros((3, 4, 5))
    token_indices = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="Batch size"):
        gather_token_activations(activations, token_indices)


def test_gather_token_activations_rejects_non_integer_indices() -> None:
    """Token indices must use an integer dtype."""
    activations = torch.zeros((2, 3, 4))
    token_indices = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="integer dtype"):
        gather_token_activations(activations, token_indices)


def test_gather_token_activations_accepts_integer_index_dtypes() -> None:
    """Integer index tensors should be accepted regardless of integer width."""
    activations = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    token_indices = torch.tensor([1, 0], dtype=torch.int32)

    selected = gather_token_activations(activations, token_indices)

    expected = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    torch.testing.assert_close(selected, expected)


@pytest.mark.parametrize(
    "token_indices",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
    ],
)
def test_gather_token_activations_rejects_out_of_bounds_indices(
    token_indices: torch.Tensor,
) -> None:
    """Token indices outside the sequence bounds should raise an error."""
    activations = torch.zeros((2, 3, 4))

    with pytest.raises(ValueError, match="sequence bounds"):
        gather_token_activations(activations, token_indices)









@pytest.mark.parametrize(
    ("attention_mask", "expected"),
    [
        (
            torch.tensor(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                ]
            ),
            torch.tensor([2, 3]),
        ),
        (
            torch.tensor(
                [
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                ]
            ),
            torch.tensor([4, 4]),
        ),
        (
            torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
            torch.tensor([3, 3]),
        ),
    ],
)
def test_get_last_non_padding_token_indices_returns_expected_indices(
    attention_mask: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Last non-padding positions should be found for different padding layouts."""
    indices = get_last_non_padding_token_indices(attention_mask)

    torch.testing.assert_close(indices, expected)


def test_get_last_non_padding_token_indices_rejects_non_2d_mask() -> None:
    """Attention masks must have batch and sequence dimensions."""
    attention_mask = torch.tensor([1, 1, 0])

    with pytest.raises(ValueError, match="two-dimensional"):
        get_last_non_padding_token_indices(attention_mask)


def test_get_last_non_padding_token_indices_rejects_non_binary_mask() -> None:
    """Attention masks may contain only binary values."""
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 2, 0],
        ]
    )

    with pytest.raises(ValueError, match="only 0 and 1"):
        get_last_non_padding_token_indices(attention_mask)


def test_get_last_non_padding_token_indices_rejects_empty_sequence() -> None:
    """Every sequence must contain at least one real token."""
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
        ]
    )

    with pytest.raises(ValueError, match="at least one non-padding token"):
        get_last_non_padding_token_indices(attention_mask)


def test_get_last_non_padding_token_indices_handles_single_item_batch() -> None:
    """A batch containing one sequence should be supported."""
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
        ]
    )

    indices = get_last_non_padding_token_indices(attention_mask)

    expected = torch.tensor([1])
    torch.testing.assert_close(indices, expected)





def test_gather_multiple_token_activations_returns_expected_vectors() -> None:
    """Multiple token positions should be selected for each batch item."""
    activations = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            [
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
        ]
    )
    token_indices = torch.tensor(
        [
            [0, 2],
            [1, 2],
        ]
    )

    selected = gather_multiple_token_activations(
        activations,
        token_indices,
    )

    expected = torch.tensor(
        [
            [
                [1.0, 2.0],
                [5.0, 6.0],
            ],
            [
                [9.0, 10.0],
                [11.0, 12.0],
            ],
        ]
    )

    torch.testing.assert_close(selected, expected)


def test_gather_multiple_token_activations_rejects_non_3d_activations() -> None:
    """Activations must include batch, sequence, and hidden dimensions."""
    activations = torch.zeros((3, 4))
    token_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )

    with pytest.raises(ValueError, match="three-dimensional"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )


def test_gather_multiple_token_activations_rejects_non_2d_indices() -> None:
    """Multiple-token indices must have batch and position dimensions."""
    activations = torch.zeros((2, 3, 4))
    token_indices = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="two-dimensional"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )


def test_gather_multiple_token_activations_rejects_batch_mismatch() -> None:
    """Token-index batch size must match the activation batch size."""
    activations = torch.zeros((3, 4, 5))
    token_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
        ]
    )

    with pytest.raises(ValueError, match="batch size"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )


def test_gather_multiple_token_activations_rejects_no_positions() -> None:
    """At least one token position must be requested."""
    activations = torch.zeros((2, 3, 4))
    token_indices = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="At least one token position"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )


def test_gather_multiple_token_activations_rejects_non_integer_indices() -> None:
    """Token positions must use an integer dtype."""
    activations = torch.zeros((2, 3, 4))
    token_indices = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="integer dtype"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )


@pytest.mark.parametrize(
    "token_indices",
    [
        torch.tensor(
            [
                [-1, 1],
                [0, 2],
            ]
        ),
        torch.tensor(
            [
                [0, 1],
                [2, 3],
            ]
        ),
    ],
)
def test_gather_multiple_token_activations_rejects_out_of_bounds_indices(
    token_indices: torch.Tensor,
) -> None:
    """Every requested token position must lie within the sequence."""
    activations = torch.zeros((2, 3, 4))

    with pytest.raises(ValueError, match="sequence bounds"):
        gather_multiple_token_activations(
            activations,
            token_indices,
        )







class _FakeModel:
    """Minimal model double used to test hidden-state extraction."""

    def __init__(
        self,
        *,
        return_hidden_states: bool = True,
    ) -> None:
        """Initialize the fake model."""
        self.return_hidden_states = return_hidden_states
        self.last_kwargs: dict[str, object] | None = None
        self.inference_mode_enabled = False

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        """Record the forward call and return deterministic model outputs."""
        self.last_kwargs = kwargs
        self.inference_mode_enabled = torch.is_inference_mode_enabled()

        if not self.return_hidden_states:
            return SimpleNamespace(hidden_states=None)

        hidden_states = (
            torch.zeros((2, 3, 4)),
            torch.ones((2, 3, 4)),
            torch.full((2, 3, 4), 2.0),
        )

        return SimpleNamespace(hidden_states=hidden_states)



def test_extract_hidden_states_returns_model_hidden_states() -> None:
    """All hidden states returned by the model should be preserved."""
    model = _FakeModel()

    input_ids = torch.tensor(
        [
            [1, 2, 0],
            [3, 4, 5],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    hidden_states = extract_hidden_states(
        model,
        model_inputs,
    )

    assert len(hidden_states) == 3
    assert hidden_states[0].shape == (2, 3, 4)

    torch.testing.assert_close(
        hidden_states[0],
        torch.zeros((2, 3, 4)),
    )
    torch.testing.assert_close(
        hidden_states[1],
        torch.ones((2, 3, 4)),
    )
    torch.testing.assert_close(
        hidden_states[2],
        torch.full((2, 3, 4), 2.0),
    )

def test_extract_hidden_states_requests_hidden_states() -> None:
    """The forward pass should explicitly request hidden states."""
    model = _FakeModel()

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])

    extract_hidden_states(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    assert model.last_kwargs is not None
    assert model.last_kwargs["input_ids"] is input_ids
    assert model.last_kwargs["attention_mask"] is attention_mask
    assert model.last_kwargs["output_hidden_states"] is True
    assert model.last_kwargs["return_dict"] is True


def test_extract_hidden_states_uses_inference_mode() -> None:
    """The model forward pass should run without autograd tracking."""
    model = _FakeModel()

    extract_hidden_states(
        model,
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    )

    assert model.inference_mode_enabled is True


def test_extract_hidden_states_rejects_empty_inputs() -> None:
    """At least one model input must be provided."""
    model = _FakeModel()

    with pytest.raises(ValueError, match="At least one model input"):
        extract_hidden_states(model, {})


def test_extract_hidden_states_rejects_missing_hidden_states() -> None:
    """Models that do not return hidden states should raise an error."""
    model = _FakeModel(return_hidden_states=False)

    with pytest.raises(RuntimeError, match="did not return hidden states"):
        extract_hidden_states(
            model,
            {
                "input_ids": torch.tensor([[1, 2]]),
            },
        )