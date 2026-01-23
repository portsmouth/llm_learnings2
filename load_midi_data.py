import os
import json
import torch
import numpy as np
from pathlib import Path


def load_vocabulary(vocab_path='vocab.json'):
    """Load the vocabulary mapping from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_single_file(file_path):
    """Load a single tokenized integer file and return as list of integers."""
    with open(file_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return []
        tokens = [int(x) for x in content.split()]
    return tokens


def load_all_midi_data(data_dir='tokenized_midi_int', max_files=None):
    """
    Load all tokenized MIDI files from the directory.

    Args:
        data_dir: Directory containing tokenized integer files
        max_files: Optional limit on number of files to load (for testing)

    Returns:
        List of token sequences (each sequence is a list of integers)
    """
    all_sequences = []
    file_count = 0

    print(f"Loading tokenized MIDI data from {data_dir}...")

    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith('_int.txt'):
                file_path = os.path.join(root, file)
                tokens = load_single_file(file_path)

                if tokens:  # Only add non-empty sequences
                    all_sequences.append(tokens)
                    file_count += 1

                    if file_count % 50 == 0:
                        print(f"  Loaded {file_count} files...")

                    if max_files and file_count >= max_files:
                        break

        if max_files and file_count >= max_files:
            break

    print(f"Loaded {file_count} files with {sum(len(s) for s in all_sequences)} total tokens")
    return all_sequences


def concatenate_sequences(sequences, add_separator=True, separator_token=0):
    """
    Concatenate all sequences into a single sequence.

    Args:
        sequences: List of token sequences
        add_separator: Whether to add separator token between sequences
        separator_token: Token ID to use as separator (default: 0 = <END>)

    Returns:
        Single concatenated list of tokens
    """
    if not sequences:
        return []

    if not add_separator:
        # Simple concatenation
        return [token for seq in sequences for token in seq]

    # Concatenate with separator tokens between sequences
    result = []
    for i, seq in enumerate(sequences):
        result.extend(seq)
        # Add separator after each sequence (including the last one)
        # This ensures every piece ends with END token
        if i < len(sequences) - 1 or add_separator:
            result.append(separator_token)

    return result


def load_midi_dataset(
    data_dir='tokenized_midi_int',
    vocab_path='vocab.json',
    train_ratio=0.9,
    add_separators=True,
    device='cpu',
    max_files=None
):
    """
    Load and prepare the MIDI dataset for training.

    Args:
        data_dir: Directory containing tokenized integer files
        vocab_path: Path to vocabulary JSON file
        train_ratio: Ratio of data to use for training (rest is validation)
        add_separators: Whether to add separator tokens between sequences
        device: Device to place tensors on ('cpu' or 'cuda')
        max_files: Optional limit on number of files to load

    Returns:
        Dictionary containing:
            - train_data: Training data tensor
            - val_data: Validation data tensor
            - vocab: Vocabulary mapping
            - vocab_size: Size of vocabulary
            - total_tokens: Total number of tokens
    """
    # Load vocabulary
    vocab = load_vocabulary(vocab_path)
    vocab_size = len(vocab)

    print("="*60)
    print("MIDI Dataset Loading")
    print("="*60)
    print(f"Vocabulary size: {vocab_size}")

    # Load all sequences
    sequences = load_all_midi_data(data_dir, max_files=max_files)

    # Concatenate all sequences
    print(f"\nConcatenating {len(sequences)} sequences...")
    all_tokens = concatenate_sequences(sequences, add_separator=add_separators)
    total_tokens = len(all_tokens)

    print(f"Total tokens: {total_tokens:,}")

    # Convert to tensor
    data = torch.tensor(all_tokens, dtype=torch.long, device=device)

    # Split into train and validation
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"\nData split:")
    print(f"  Training tokens: {len(train_data):,} ({train_ratio*100:.0f}%)")
    print(f"  Validation tokens: {len(val_data):,} ({(1-train_ratio)*100:.0f}%)")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'total_tokens': total_tokens
    }


def get_batch(data, block_size, batch_size, device='cpu'):
    """
    Generate a batch of training data.

    Args:
        data: Full dataset tensor
        block_size: Context length (number of tokens in sequence)
        batch_size: Number of sequences in batch
        device: Device to place tensors on

    Returns:
        x: Input sequences (batch_size, block_size)
        y: Target sequences (batch_size, block_size)
    """
    # Generate random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Extract sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == '__main__':
    """
    Test the data loading functionality.
    """
    print("Testing MIDI data loader...")
    print()

    # Load dataset
    dataset = load_midi_dataset(
        data_dir='tokenized_midi_int',
        vocab_path='vocab.json',
        train_ratio=0.9,
        add_separators=True,
        device='cpu'
    )

    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Vocabulary size: {dataset['vocab_size']}")
    print(f"Total tokens: {dataset['total_tokens']:,}")
    print(f"Training tokens: {len(dataset['train_data']):,}")
    print(f"Validation tokens: {len(dataset['val_data']):,}")

    # Test batch generation
    print("\n" + "="*60)
    print("Testing Batch Generation")
    print("="*60)
    block_size = 256
    batch_size = 4

    xb, yb = get_batch(dataset['train_data'], block_size, batch_size)

    print(f"Input batch shape: {xb.shape}")
    print(f"Target batch shape: {yb.shape}")
    print(f"\nFirst sequence (first 20 tokens):")
    print(f"  Input:  {xb[0][:20].tolist()}")
    print(f"  Target: {yb[0][:20].tolist()}")

    print("\nData loader test complete!")
