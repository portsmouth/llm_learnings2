"""
Example training loop for MIDI music generation using the tokenized dataset.
This demonstrates how to use the data loader for actual training.
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from load_midi_data import load_midi_dataset, get_batch
from decode_midi import tokens_to_midi, play_midi
from models import SimpleTransformer
from dotenv import load_dotenv

# read config into env variables
load_dotenv()

dataset = os.getenv("DATASET")
datasets_dir = "./datasets"
dataset_dir = f"{datasets_dir}/{dataset}"
dataset_input_dir = f"{dataset_dir}/data"
tokenized_output_dir = f"{dataset_dir}/tokenized_midi"
tokenized_ints_output_dir = f"{dataset_dir}/tokenized_midi_int"
vocab_path = f"{dataset_dir}/vocab.json"


# Set random seed for reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 64
block_size = 256  # Context length
max_iters = 5000
head_size = 64
n_embed = 256
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

print(f"Using device: {device}")
print("="*60)

# Load MIDI dataset
dataset = load_midi_dataset(
    data_dir=tokenized_ints_output_dir,
    vocab_path=vocab_path,
    train_ratio=0.9,
    add_separators=True,
    device=device
)

train_data = dataset['train_data']
val_data = dataset['val_data']
vocab_size = dataset['vocab_size']

print("="*60)


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

model = SimpleTransformer(vocab_size, block_size, head_size, n_embed).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nStarting training...")
print("="*60)

# Training loop
for iter in range(max_iters):

    # Sample a batch of data
    xb, yb = get_batch(train_data, block_size, batch_size, device)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Evaluate periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)

# Generate sample output
print("\nGenerating sample output...")
end_token_id = dataset['vocab'].get('<END>', None)
if end_token_id is not None:
    context = torch.tensor([[end_token_id]], dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
if end_token_id:
    print(f"Model can stop generation early if <END> token (ID={end_token_id}) is sampled")
generated = model.generate(context, max_new_tokens=500, end_token_id=end_token_id)
print(f"\nGenerated sequence ({generated.shape[1]} tokens):")
print(generated[0].tolist()[:50], "...")  # Show first 50 tokens

# Save the model
torch.save(model.state_dict(), 'midi_model.pth')
print("\nModel saved to midi_model.pth")

# Decode generated sequence to MIDI
print("\n" + "="*60)
print("Converting generated output to MIDI...")
print("="*60)
midi_path = tokens_to_midi(generated[0].cpu(), 'generated_music.mid')

# Play the generated MIDI file
print("\nAttempting to play generated MIDI...")
play_midi(midi_path)

print("\n" + "="*60)
print(f"Generated MIDI file: {midi_path}")
print("You can also open this file in any MIDI player or DAW")
print("="*60)

