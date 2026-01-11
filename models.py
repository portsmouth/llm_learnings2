"""
Neural network models for MIDI music generation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """
    Simple bigram language model for MIDI generation.

    This model uses a simple embedding table to predict the next token
    based on the current token, making it a bigram model.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)

        Returns:
            logits: (B, T, vocab_size) tensor of logits
            loss: scalar loss value (if targets provided, else None)
        """
        # idx: (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, end_token_id=None):
        """
        Generate new tokens from the model.

        Args:
            idx: (B, T) tensor of starting context indices
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random, lower = more conservative)
            top_k: If set, only sample from top k most likely tokens
            end_token_id: If set, stop generation when this token is sampled

        Returns:
            (B, T+max_new_tokens) tensor of generated token indices
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            # Focus on last time step
            logits = logits[:, -1, :]  # (B, C)

            # Apply temperature
            logits = logits / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            # Stop if END token is generated
            if end_token_id is not None and idx_next.item() == end_token_id:
                break

        return idx
