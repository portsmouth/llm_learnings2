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


###############################################################



class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, block_size, n_embed):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X)      # (B, T, head_size)
        q = self.query(X)    # (B, T, head_size)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(X) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, block_size, n_embed, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple feed-forward neural network"""

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, block_size, n_embed, n_head=4, dropout=0.0, use_ffwd=True):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"
        head_size = n_embed // n_head
        self.sa_head = MultiHeadAttention(num_heads=n_head, head_size=head_size, block_size=block_size, n_embed=n_embed, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.use_ffwd = use_ffwd
        if self.use_ffwd:
            self.ffwd = FeedForward(n_embed, dropout=dropout)
            self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, X):
        X = X + self.sa_head(self.ln1(X))
        if self.use_ffwd:
            X = X + self.ffwd(self.ln2(X))
        return X




class SimpleTransformer(nn.Module):
    """
    Transformer language model for MIDI generation.

    Architecture follows nanoGPT style with:
    - Token + position embeddings
    - N transformer blocks with multi-head attention and feedforward
    - Final layer normalization
    - LM head for next token prediction
    """

    def __init__(self, vocab_size, block_size, n_embed, n_layer=3, n_head=4, dropout=0.0, use_ffwd=True):
        """
        Initialize the language model.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            block_size (int): Maximum context length (sequence length).
            n_embed (int): Embedding dimension size.
            n_layer (int): Number of transformer blocks (default: 3).
            n_head (int): Number of attention heads (default: 4).
            dropout (float): Dropout probability (default: 0.0).
            use_ffwd (bool): Whether to include feedforward layers (default: True).
        """
        super().__init__()

        self.n_embed = n_embed
        self.block_size = block_size
        self.n_head = n_head
        self.use_ffwd = use_ffwd

        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(block_size, n_embed, n_head=n_head, dropout=dropout, use_ffwd=use_ffwd) for _ in range(n_layer)]
        )

        # Final layer norm (nanoGPT style)
        self.ln_f = nn.LayerNorm(n_embed)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(n_embed, vocab_size)


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
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=idx.device))  # (T, C)

        X = tok_emb + pos_emb  # (B, T, C)
        X = self.dropout(X)  # Apply dropout to embeddings

        X = self.blocks(X)  # (B, T, C)
        X = self.ln_f(X)  # Final layer norm (B, T, C)

        logits = self.lm_head(X)  # (B, T, vocab_size)

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

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits, loss = self(idx_cond)

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