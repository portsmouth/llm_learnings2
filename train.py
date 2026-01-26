import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from load_midi_data import load_midi_dataset, get_batch
from dotenv import load_dotenv

# read config into env variables
load_dotenv()

dataset = os.getenv("DATASET")
datasets_dir = "./datasets"
dataset_dir = f"{datasets_dir}/{dataset}"
dataset_input_dir = f"{dataset_dir}/data"
tokenized_output_dir = f"{dataset_dir}/tokenized_midi"
tokenized_ints_output_dir = f"{dataset_dir}/tokenized_midi_int"


# Set random seed for reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 32
block_size = 128  # Context length
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print("="*60)

# Load MIDI dataset
dataset = load_midi_dataset(
    data_dir=f"{tokenized_ints_output_dir}",
    vocab_path='vocab.json',
    train_ratio=0.9,
    add_separators=True,
    device=device
)

train_data = dataset['train_data']
val_data = dataset['val_data']
vocab_size = dataset['vocab_size']

print("="*60)
print("Training started")
print("="*60)

# Example: Show how data looks
print(f"\nFirst 20 tokens from training data:")
print(train_data[:20].tolist())

# Generate a sample batch
def get_train_batch():
    return get_batch(train_data, block_size, batch_size, device)

def get_val_batch():
    return get_batch(val_data, block_size, batch_size, device)

xb, yb = get_train_batch()
print(f"\nBatch shapes:")
print(f"  Input (xb): {xb.shape}")
print(f"  Target (yb): {yb.shape}")


# Model definition
class BigramLanguageModeler(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        return logits


# Initialize model
print(f"\nInitializing model with vocab_size={vocab_size}")
model = BigramLanguageModeler(vocab_size).to(device)
out = model(xb, yb)
print(f"Model output shape: {out.shape}")

print("\n" + "="*60)
print("Setup complete! Ready for training.")
print("="*60)
