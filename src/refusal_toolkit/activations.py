"""Utilities for selecting and processing model activations."""

import torch


def get_last_non_padding_token_indices(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Find the last non-padding token position in each sequence.

    Args:
        attention_mask: Binary attention mask with shape
            ``(batch_size, sequence_length)``, where ``1`` marks a real token
            and ``0`` marks padding.

    Returns:
        Index of the last non-padding token in each sequence, with shape
        ``(batch_size,)``.

    Raises:
        ValueError: If ``attention_mask`` is not two-dimensional.
        ValueError: If the attention mask contains values other than ``0`` or ``1``.
        ValueError: If any sequence contains no non-padding tokens.
    """
    if attention_mask.ndim != 2:
        raise ValueError("Attention mask must be two-dimensional.")

    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Attention mask must contain only 0 and 1.")

    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("Each sequence must contain at least one non-padding token.")

    sequence_length = attention_mask.shape[1]

    positions = torch.arange(   
        sequence_length,   #tüm batchdei data için 0,1,2,3,...,sequence_length-1 şeklinde bir tensor oluşturur
        device=attention_mask.device,
    ).expand_as(attention_mask)

    masked_positions = torch.where(
        attention_mask.bool(),
        positions,   #positionsun padding pozisyonlarını -1 ile değiştirir. attention_mask bool olduğunda 1 olan yerler positionsun kendisi olur , 0 olan yerler -1 olur
        -1,            # mesela [1,1,1,0,0] attention mask için (son 2 si padding) masked position[0,1,2,-1,-1] olur
    )

    return masked_positions.max(dim=1).values   # masked olmayan en büyük index son paddingdir

def gather_token_activations(
    activations: torch.Tensor,
    token_indices: torch.Tensor,   # mesela [1,3,5] demek 1. sampledan 1. tokenı al, 2. sapmledan 2. tokenı al.. o yüzden batch sizela aynı olmalı
) -> torch.Tensor:
    """Select one token activation from each sequence in a batch.

    Args:
        activations: Activation tensor with shape
            ``(batch_size, sequence_length, hidden_dim)``.
        token_indices: Token position for each sequence, with shape
            ``(batch_size,)``.   

    Returns:
        Selected activation vectors with shape
        ``(batch_size, hidden_dim)``.

    Raises:
        ValueError: If ``activations`` is not three-dimensional.
        ValueError: If ``token_indices`` is not one-dimensional.
        ValueError: If the batch dimensions do not match.
        ValueError: If any token index is outside the sequence bounds.
    """
    if activations.ndim != 3:
        raise ValueError("Activations must be three-dimensional.")

    if token_indices.ndim != 1:
        raise ValueError("Token indices must be one-dimensional.")

    if activations.shape[0] != token_indices.shape[0]:
        raise ValueError("Batch size must match the number of token indices.")

    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    if token_indices.dtype not in integer_dtypes:
        raise ValueError("Token indices must use an integer dtype.")

    sequence_length = activations.shape[1]

    if torch.any(token_indices < 0) or torch.any(token_indices >= sequence_length):
        raise ValueError("Token indices must be within the sequence bounds.")

    token_indices = token_indices.to(
        device=activations.device,
        dtype=torch.long,
    )

    batch_indices = torch.arange(   #eğer batch size 3 ise, bu [0,1,2] olur
        activations.shape[0],
        device=activations.device,
    )

    return activations[batch_indices, token_indices] #sonra sırayla [0 dan, 1. tokenın hidden dimensionu nu al, 1.den 3. tokenın hidden dimensounun al..] output da batchsize x hidden dimension olur