"""
Fine-tune existing model with style conditioning.
Starts from a pre-trained model that already generates good music,
then adds style awareness.
"""

import torch
import torch.nn as nn
from load_midi_data_with_style import load_midi_dataset_with_style, get_batch
from models import SimpleTransformer
import time
import math

# Set random seed
torch.manual_seed(1337)

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 10000  # More iterations for fine-tuning
eval_interval = 250
eval_iters = 200
log_interval = 10

# Fine-tuning settings (lower learning rate!)
learning_rate = 1e-4  # Much lower than from-scratch training
min_lr = 1e-5
warmup_iters = 100
lr_decay_iters = 10000
grad_clip = 1.0

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print("="*60)

# Load MIDI dataset with style conditioning
dataset = load_midi_dataset_with_style(
    data_dir='tokenized_midi_int',
    labels_path='midi_labels.json',
    vocab_path='vocab.json',
    train_ratio=0.9,
    add_separators=True,
    device=device
)

train_data = dataset['train_data']
val_data = dataset['val_data']
vocab_size = dataset['vocab_size']
vocab = dataset['vocab']

print("="*60)

# Load pre-trained model
print("\nLoading pre-trained model from midi_model.pth...")
try:
    checkpoint = torch.load('midi_model.pth', map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}

    # Get model architecture from config
    n_embed = config.get('n_embed', 256)
    n_layer = config.get('n_layer', 3)
    n_head = config.get('n_head', 4)
    old_vocab_size = config.get('vocab_size', 780)

    print(f"  Pre-trained model config: vocab={old_vocab_size}, embed={n_embed}, layers={n_layer}, heads={n_head}")

    # Create new model with EXPANDED vocabulary (to include style tokens)
    model = SimpleTransformer(vocab_size, block_size, n_embed,
                             n_layer=n_layer, n_head=n_head, dropout=0.2).to(device)

    # Load weights, but expand embedding layers for new tokens
    model_dict = model.state_dict()

    # Copy all weights except embedding layers
    for key in state_dict:
        if key in model_dict:
            if 'embedding' not in key and 'lm_head' not in key:
                # Copy non-embedding weights directly
                model_dict[key] = state_dict[key]
            elif key == 'token_embedding_table.weight':
                # Copy old embeddings, initialize new ones randomly
                old_emb = state_dict[key]
                new_emb = model_dict[key]
                # Copy embeddings for existing tokens
                new_emb[:old_vocab_size] = old_emb
                # New style tokens get random initialization
                print(f"  Expanded token embeddings: {old_vocab_size} -> {vocab_size}")
            elif key == 'lm_head.weight':
                # Expand output projection
                old_head = state_dict[key]
                new_head = model_dict[key]
                new_head[:old_vocab_size] = old_head
                print(f"  Expanded output head: {old_vocab_size} -> {vocab_size}")

    model.load_state_dict(model_dict)
    print("  Pre-trained weights loaded successfully!")

except FileNotFoundError:
    print("  No pre-trained model found, training from scratch...")
    n_embed = 256
    n_layer = 3
    n_head = 4
    model = SimpleTransformer(vocab_size, block_size, n_embed,
                             n_layer=n_layer, n_head=n_head, dropout=0.2).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
def get_lr(iter):
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > lr_decay_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                train_data if split == 'train' else val_data,
                block_size,
                batch_size,
                device
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\nFine-tuning with style conditioning...")
print("="*60)

# Training loop
t0 = time.time()

for iter in range(max_iters):
    # Set learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        if iter > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter,
                'config': {
                    'vocab_size': vocab_size,
                    'block_size': block_size,
                    'n_embed': n_embed,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'dropout': 0.2,
                }
            }
            torch.save(checkpoint, f'ckpt_finetune_{iter}.pt')

    # Logging
    if iter % log_interval == 0 and iter > 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item()
        print(f"iter {iter}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")

    # Sample batch
    xb, yb = get_batch(train_data, block_size, batch_size, device)

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Clip gradients
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update weights
    optimizer.step()

print("\n" + "="*60)
print("Fine-tuning complete!")
print("="*60)

# Save final model
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter': max_iters,
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embed': n_embed,
        'n_layer': n_layer,
        'n_head': n_head,
        'dropout': 0.2,
    }
}
torch.save(checkpoint, 'midi_model_finetuned.pth')
print("\nModel saved to midi_model_finetuned.pth")
print("Test with: python generate_with_style.py --model midi_model_finetuned.pth --prompt \"fast romantic\"")
