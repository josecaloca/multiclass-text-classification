"""
Module for extracting embeddings from a transformer-based model.

This module provides a function to extract sentence embeddings from a batch of tokenized
inputs using a pre-trained transformer model. It applies mean pooling over the last hidden 
state to obtain fixed-size embeddings.
"""

import torch
from torch import nn
from typing import Dict, Any


def extract_embeddings(batch: Dict[str, Any], model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts sentence embeddings from a batch of tokenized inputs using a transformer model.

    The function:
    - Converts input IDs and attention masks to PyTorch tensors.
    - Pads sequences to ensure uniform length within the batch.
    - Passes the inputs through the model in inference mode (no gradient computation).
    - Computes mean pooling over the last hidden state to obtain embeddings.

    Args:
        batch (Dict[str, Any]): A dictionary containing 'input_ids' and 'attention_mask',
                                each being a list of tokenized sequences.
        model (nn.Module): A pre-trained transformer model for embedding extraction.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the extracted embeddings with the key 'embeddings'.
    """
    input_ids = [torch.tensor(seq) for seq in batch["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in batch["attention_mask"]]

    # Pad sequences to ensure uniform shape
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    with torch.no_grad():  # Disable gradient computation for efficiency
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Apply mean pooling over the last hidden state to obtain sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    return {"embeddings": embeddings}
