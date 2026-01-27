"""
Training loop for MIDI music generation with style conditioning.
Uses style tokens prepended to sequences for controllable generation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from load_midi_data_with_style import load_midi_dataset_with_style, get_batch
from decode_midi import tokens_to_midi, play_midi
from models import SimpleTransformer
import time
import math

# Set random seed for reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 64
block_size = 256  # Context length
max_iters = 10000

n_embed = 384  # Embedding dimension
n_layer = 6  # Number of transformer blocks
n_head = 4  # Number of attention heads (n_embed must be divisible by n_head)
dropout = 0.2  # Dropout probability (0.0 = no dropout, 0.2 = 20% dropout)
eval_interval = 250
eval_iters = 200
log_interval = 10  # Log every N iterations

# Optimizer settings
learning_rate = 6e-4  # Max learning rate
min_lr = 6e-5  # Min learning rate (for cosine decay)
warmup_iters = 100  # Warmup iterations
lr_decay_iters = 5000  # Should be ~= max_iters
grad_clip = 1.0  # Clip gradients at this value

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile_model = False  # Set to True if using PyTorch 2.0+

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


# Learning rate scheduler (cosine with warmup)
def get_lr(iter):
    # 1) Linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) If iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
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


# Initialize model
print("Initializing model...")

model = SimpleTransformer(vocab_size, block_size, n_embed, n_layer=n_layer, n_head=n_head, dropout=dropout).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Compile model for better performance (PyTorch 2.0+)
if compile_model:
    print("Compiling model... (takes a ~minute)")
    model = torch.compile(model)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nStarting training with style conditioning...")
print("="*60)

# Training loop
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

for iter in range(max_iters):

    # Set learning rate for this iteration
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
                    'dropout': dropout,
                }
            }
            torch.save(checkpoint, f'ckpt_style_{iter}.pt')

    # Logging
    if iter % log_interval == 0 and iter > 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item()
        print(f"iter {iter}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")

    # Sample a batch of data
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
print("Training complete!")
print("="*60)

# Generate sample output with style conditioning
print("\nGenerating sample output with style: ROMANTIC + SLOW + CALM...")

# Create context with style tokens
style_tokens = ['<STYLE_ROMANTIC>', '<TEMPO_SLOW>', '<MOOD_CALM>', '<DYN_SOFT>']
style_ids = [vocab[token] for token in style_tokens if token in vocab]
context = torch.tensor([style_ids], dtype=torch.long, device=device)

print(f"Style context: {style_tokens}")
print(f"Style token IDs: {style_ids}")

# Generate
end_token_id = vocab.get('<END>', None)
generated = model.generate(context, max_new_tokens=500, end_token_id=end_token_id)

print(f"\nGenerated sequence ({generated.shape[1]} tokens):")
print(generated[0].tolist()[:50], "...")  # Show first 50 tokens

# Save the final model
print("\nSaving final model...")
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
        'dropout': dropout,
    }
}
torch.save(checkpoint, 'midi_model_style.pth')
print("Model saved to midi_model_style.pth")

# Decode generated sequence to MIDI
print("\n" + "="*60)
print("Converting generated output to MIDI...")
print("="*60)
midi_path = tokens_to_midi(generated[0].cpu(), 'generated_music_style.mid')

# Play the generated MIDI file
print("\nAttempting to play generated MIDI...")
play_midi(midi_path)

print("\n" + "="*60)
print(f"Generated MIDI file: {midi_path}")
print("You can also open this file in any MIDI player or DAW")
print("="*60)
