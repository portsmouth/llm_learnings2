# SimpleTransformer Architecture

## Data Flow Diagram

```
INPUT TOKENS
    │
    │ idx: (B, T)
    │ B = batch_size (e.g., 64)
    │ T = sequence_length (≤ block_size, e.g., 256)
    │
    └──────────────────────────────────────────────────────────┐
                                                                │
    ┌───────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────┐
    │                         │
    ▼                         ▼
TOKEN EMBEDDING          POSITION EMBEDDING
    │                         │
    │ token_embedding          │ position_embedding
    │ table                    │ table
    │                         │
    │ (B, T, n_embed)          │ (T, n_embed)
    │ n_embed = 256            │ n_embed = 256
    │                         │
    └─────────────┬───────────┘
                  │
                  ▼
            COMBINED EMBEDDING
                  │
                  │ X = tok_emb + pos_emb
                  │ (B, T, n_embed)
                  │ (64, 256, 256)
                  │
                  ▼
        ┌─────────────────────┐
        │  MULTI-HEAD         │
        │  ATTENTION          │
        │  (4 heads)          │
        └─────────────────────┘
                  │
                  │ Each head processes:
                  │   Q = query(X)   → (B, T, head_size/4) = (64, 256, 16)
                  │   K = key(X)     → (B, T, head_size/4) = (64, 256, 16)
                  │   V = value(X)   → (B, T, head_size/4) = (64, 256, 16)
                  │
                  │   Attention weights:
                  │   wei = Q @ K^T / sqrt(head_size/4)
                  │   wei = (B, T, T) = (64, 256, 256)
                  │
                  │   Masked with lower triangular
                  │   Softmax applied
                  │
                  │   Output per head:
                  │   out_i = wei @ V → (B, T, head_size/4)
                  │
                  │ Concatenate all heads:
                  │ concat([out_1, out_2, out_3, out_4])
                  │ (B, T, head_size) = (64, 256, 64)
                  │
                  │ Project back to n_embed:
                  │ proj(concat)
                  │
                  ▼
            (B, T, n_embed)
            (64, 256, 256)
                  │
                  ▼
        ┌─────────────────────┐
        │  FEED FORWARD       │
        │  NETWORK            │
        └─────────────────────┘
                  │
                  │ Linear1: n_embed → 4*n_embed
                  │ (B, T, 256) → (B, T, 1024)
                  │
                  │ ReLU activation
                  │
                  │ Linear2: 4*n_embed → n_embed
                  │ (B, T, 1024) → (B, T, 256)
                  │
                  ▼
            (B, T, n_embed)
            (64, 256, 256)
                  │
                  ▼
        ┌─────────────────────┐
        │  LANGUAGE MODEL     │
        │  HEAD               │
        │  (Linear)           │
        └─────────────────────┘
                  │
                  │ lm_head: n_embed → vocab_size
                  │ (B, T, 256) → (B, T, 772)
                  │
                  ▼
            OUTPUT LOGITS
            (B, T, vocab_size)
            (64, 256, 772)
                  │
                  │
                  ├─── Training: Compute cross-entropy loss
                  │    Reshape to (B*T, vocab_size) = (16384, 772)
                  │    Compare with targets (B*T)
                  │
                  └─── Generation: Take last token logits
                       logits[:, -1, :] → (B, vocab_size) = (1, 772)
                       Apply temperature & top-k
                       Sample next token
```

## Key Components

### 1. Token Embedding Table
- **Input:** Token indices `(B, T)`
- **Output:** Dense vectors `(B, T, n_embed)`
- **Parameters:** `vocab_size × n_embed` = `772 × 256`

### 2. Position Embedding Table
- **Input:** Position indices `(T,)`
- **Output:** Dense vectors `(T, n_embed)`
- **Parameters:** `block_size × n_embed` = `256 × 256`
- **Note:** Broadcast added to token embeddings

### 3. Multi-Head Attention (4 heads)
- **Input:** `(B, T, n_embed)` = `(64, 256, 256)`
- **Per Head:**
  - Query, Key, Value projections: `n_embed → head_size/4` = `256 → 16`
  - Attention computation: `Q @ K^T → (B, T, T)`
  - Masked with causal mask (lower triangular)
  - Applied to Values: `wei @ V → (B, T, 16)`
- **Concatenation:** 4 heads × 16 = 64 dimensions
- **Projection:** `head_size → n_embed` = `64 → 256`
- **Output:** `(B, T, n_embed)` = `(64, 256, 256)`

### 4. Feed Forward Network
- **Input:** `(B, T, n_embed)` = `(64, 256, 256)`
- **Hidden Layer:** `n_embed → 4*n_embed` = `256 → 1024` (with ReLU)
- **Output Layer:** `4*n_embed → n_embed` = `1024 → 256`
- **Output:** `(B, T, n_embed)` = `(64, 256, 256)`

### 5. Language Model Head
- **Input:** `(B, T, n_embed)` = `(64, 256, 256)`
- **Linear:** `n_embed → vocab_size` = `256 → 772`
- **Output:** `(B, T, vocab_size)` = `(64, 256, 772)`

## Example with Real Numbers

Given typical hyperparameters:
- `batch_size = 64`
- `block_size = 256` (max context length)
- `vocab_size = 772` (number of unique tokens)
- `n_embed = 256` (embedding dimension)
- `head_size = 64` (total across 4 heads, 16 per head)

### Forward Pass Trace

```python
# Input
idx = torch.tensor([[...]])  # Shape: (64, 256)

# Token + Position Embeddings
tok_emb = token_embedding_table(idx)      # (64, 256, 256)
pos_emb = position_embedding_table(...)   # (256, 256)
X = tok_emb + pos_emb                     # (64, 256, 256)

# Multi-Head Attention
# - Each of 4 heads: (64, 256, 256) → (64, 256, 16)
# - Concatenated: (64, 256, 64)
# - Projected: (64, 256, 256)
X = sa_head(X)                            # (64, 256, 256)

# Feed Forward
# - Expand: (64, 256, 256) → (64, 256, 1024)
# - Contract: (64, 256, 1024) → (64, 256, 256)
X = ffwd(X)                               # (64, 256, 256)

# Output Projection
logits = lm_head(X)                       # (64, 256, 772)

# Loss Computation (training)
logits_flat = logits.view(16384, 772)    # Flatten batch & sequence
targets_flat = targets.view(16384)       # Flatten targets
loss = cross_entropy(logits_flat, targets_flat)
```

## Generation Process

During generation, the model processes sequences autoregressively:

```
Generation Loop:
  1. Start with context: (1, initial_length)
  2. Forward pass → logits: (1, initial_length, 772)
  3. Take last token logits: logits[:, -1, :] → (1, 772)
  4. Apply temperature: logits / temperature
  5. Optionally apply top-k filtering
  6. Sample: multinomial(softmax(logits)) → (1, 1)
  7. Append to context: (1, initial_length+1)
  8. Crop context to block_size if needed: [:, -block_size:]
  9. Repeat until max_tokens or <END> token
```

## Parameter Count

```
Token Embedding:      772 × 256 = 197,632
Position Embedding:   256 × 256 = 65,536

Multi-Head Attention (per head):
  Query:  256 × 16 = 4,096
  Key:    256 × 16 = 4,096
  Value:  256 × 16 = 4,096
  (× 4 heads = 49,152)

Attention Projection: 64 × 256 + 256 = 16,640

Feed Forward:
  Linear1: 256 × 1024 + 1024 = 263,168
  Linear2: 1024 × 256 + 256 = 262,400

LM Head:             256 × 772 + 772 = 198,404

Total: ~1.06M parameters
```

## Notes

- **Causal Masking:** The attention mechanism uses a lower triangular mask to ensure tokens only attend to previous positions, maintaining the autoregressive property.

- **Context Window:** The model can only look back `block_size` tokens. During generation, longer sequences are cropped to the last `block_size` tokens.

- **No Residual Connections:** This simple version doesn't include residual connections or layer normalization, which are common in larger transformers.

- **Single Layer:** This is a single-layer transformer. Deeper models would repeat the attention + feedforward blocks multiple times.
