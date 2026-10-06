"""Utilities for selecting and processing model activations."""

from collections.abc import Mapping

import torch
from transformers import PreTrainedModel


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


def gather_multiple_token_activations(  #for zhao type of experriments
    activations: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Select multiple token activations from each sequence in a batch.

    Args:
        activations: Activation tensor with shape
            ``(batch_size, sequence_length, hidden_dim)``.
        token_indices: Token positions for each sequence, with shape
            ``(batch_size, num_positions)``.  

    Returns:
        Selected activation vectors with shape
        ``(batch_size, num_positions, hidden_dim)``.

    Raises:
        ValueError: If ``activations`` is not three-dimensional.
        ValueError: If ``token_indices`` is not two-dimensional.
        ValueError: If the batch dimensions do not match.
        ValueError: If no token positions are provided.
        ValueError: If ``token_indices`` does not use an integer dtype.
        ValueError: If any token index is outside the sequence bounds.
    """
    if activations.ndim != 3:
        raise ValueError("Activations must be three-dimensional.")

    if token_indices.ndim != 2:
        raise ValueError("Token indices must be two-dimensional.")

    if activations.shape[0] != token_indices.shape[0]:
        raise ValueError("Batch size must match the token-index batch size.")

    if token_indices.shape[1] == 0:
        raise ValueError("At least one token position must be provided.")

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

    if torch.any(token_indices < 0) or torch.any(
        token_indices >= sequence_length
    ):
        raise ValueError("Token indices must be within the sequence bounds.")

    token_indices = token_indices.to(
        device=activations.device,
        dtype=torch.long,
    )

    batch_indices = torch.arange( #bizim token indicesimiz 2D, her rowda o batchin sampleının hangi tokenlarını alacağımızı belirtiyor, 2 taneyse her rowda iki sayı mesela
        activations.shape[0],
        device=activations.device,
    ).unsqueeze(1)  #önceki fonksiyonla tel fark burada, 2D ouğu için böyle unsequeeze yapmamız lazım ki [[0],[1],[2]] gibi bir shape olsun, sonra broadcasting ile her rowdaki token indices ile eşleşecek

    return activations[batch_indices, token_indices]





def extract_hidden_states(
    model: PreTrainedModel,
    model_inputs: Mapping[str, torch.Tensor],  #tokenizationdaki encoded[input_ids], encoded[attention_mask] gibi şeyleri içeriyo
) -> tuple[torch.Tensor, ...]: #birden fazla pytorch tensor döndürebilir, embedding output, layer 0 output, layer 1 output..
    """Run a model forward pass and return all hidden states.

    Args:
        model: Hugging Face model used for the forward pass.
        model_inputs: Tensor inputs accepted by the model, such as
            ``input_ids`` and ``attention_mask``.

    Returns:
        Hidden-state tensors returned by the model. For standard Hugging Face
        transformer models, the tuple typically contains the embedding output
        followed by the output of each transformer layer.

    Raises:
        ValueError: If no model inputs are provided.
        RuntimeError: If the model does not return hidden states.
    """
    if len(model_inputs) == 0:
        raise ValueError("At least one model input must be provided.")

    with torch.inference_mode():  #do not create gradients, we are just extracting hidden states
        outputs = model(
            **model_inputs,  #input id's ve attention mask gibi şeyler içeriyo
            output_hidden_states=True,  # normalde model bize son layerın logitlerini döndürür, ama biz tüm layerların hidden state'lerini istiyoruz, embedding layer dahil
            return_dict=True,  #.ıktıyı dictionary olarak döndürür, böylece hidden states e outputs.hidden_states ile ulaşabiliriz, daha readable olur
        )

    hidden_states = getattr(outputs, "hidden_states", None) #outputs objesinin hidden_states diye bir alanı var mı? varsa direkt outputs.hidden_states olur yoksa None

    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    return tuple(hidden_states) #hidden_states = (tensor_0, tensor_1..), her tenson (batch_size, sequence_length, hidden_dimension) şeklinde, embedding layer dahil tüm layerların outputları var