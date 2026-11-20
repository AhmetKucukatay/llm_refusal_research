"""Unit tests for generic intervention-aware generation."""

from types import SimpleNamespace

import torch
from torch import nn

from refusal_toolkit.generation import generate_completions


class ToyTokenizer:
    """Minimal tokenizer interface needed by generation."""

    pad_token_id = 0

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode toy token IDs as a space-separated string."""
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


class ToyGenerator(nn.Module):
    """Minimal model implementing generate."""

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config,
    ) -> torch.Tensor:
        """Append two deterministic tokens to each prompt."""
        del attention_mask
        del generation_config

        batch_size = input_ids.shape[0]

        new_tokens = torch.tensor(
            [[7, 8]],
            device=input_ids.device,
        ).repeat(batch_size, 1)

        return torch.cat(
            [input_ids, new_tokens],
            dim=-1,
        )


def test_generate_completions_keeps_only_generated_tokens() -> None:
    """Generated records exclude the original prompt tokens."""
    #model.generate: [3 0 7 8], model length = 2, should be [7 8]
    model = ToyGenerator()
    tokenizer = ToyTokenizer()

    def tokenize(instructions: list[str]):
        input_ids = torch.tensor(
            [
                [int(instruction), 0]
                for instruction in instructions
            ]
        )

        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    records = [
        {
            "instruction": "3",
            "category": "test",
        }
    ]

    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        records=records,
        tokenize_instructions_fn=tokenize,
        max_new_tokens=2,
    )

    assert completions == [
        {
            "category": "test",
            "prompt": "3",
            "response": "7 8",
        }
    ]