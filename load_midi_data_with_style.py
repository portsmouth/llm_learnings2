"""
Data loader for MIDI dataset with style conditioning.
Extends load_midi_data.py to prepend style tokens from midi_labels.json.
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Tuple


def load_style_labels(labels_path='midi_labels.json'):
    """Load style labels from JSON file."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_midi_file_with_style(filepath: str, vocab: dict, style_labels: dict,
                               tokenized_dir: str = 'tokenized_midi_int',
                               midi_dir: str = 'kaggle_archive'):
    """
    Load a single MIDI file's tokens and prepend its style tokens.

    Args:
        filepath: Path to tokenized integer file
        vocab: Vocabulary mapping
        style_labels: Dictionary mapping MIDI paths to style tokens
        tokenized_dir: Directory containing tokenized files
        midi_dir: Original MIDI directory

    Returns:
        List of token IDs with style tokens prepended
    """
    # Read the tokenized integers
    with open(filepath, 'r') as f:
        content = f.read().strip()
        if content:
            token_ids = [int(x) for x in content.split()]
        else:
            token_ids = []

    # Find corresponding MIDI file path in labels
    # Convert tokenized path back to MIDI path
    rel_path = os.path.relpath(filepath, tokenized_dir)
    midi_rel_path = rel_path.replace('_int.txt', '.mid')

    # Normalize path separators for matching
    midi_rel_path = midi_rel_path.replace('\\', '/')

    # Find matching label entry
    style_tokens = None
    for label_path, tokens in style_labels.items():
        # Normalize the label path too
        normalized_label = label_path.replace('\\', '/')
        if normalized_label == midi_rel_path or normalized_label.endswith(midi_rel_path):
            style_tokens = tokens
            break

    # Convert style tokens to IDs
    style_ids = []
    if style_tokens:
        for token in style_tokens:
            if token in vocab:
                style_ids.append(vocab[token])

    # Prepend style tokens to sequence
    # Format: [STYLE_TOKENS] + [MIDI_TOKENS]
    if style_ids:
        return style_ids + token_ids
    else:
        return token_ids


def load_all_midi_data_with_style(data_dir='tokenized_midi_int',
                                   labels_path='midi_labels.json',
                                   vocab_path='vocab.json',
                                   max_files=None):
    """
    Load all tokenized MIDI data with style prefixes.

    Args:
        data_dir: Directory with tokenized integer files
        labels_path: Path to style labels JSON
        vocab_path: Path to vocabulary JSON
        max_files: Maximum number of files to load (None = all)

    Returns:
        List of token sequences (each with style tokens prepended)
    """
    # Load vocabulary and labels
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    style_labels = load_style_labels(labels_path)

    print(f"Loading MIDI data with style conditioning from '{data_dir}'...")
    print(f"Style labels loaded: {len(style_labels)} files")

    all_sequences = []
    file_count = 0

    # Walk through all tokenized integer files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_int.txt'):
                filepath = os.path.join(root, file)

                try:
                    sequence = load_midi_file_with_style(filepath, vocab, style_labels, data_dir)
                    if len(sequence) > 0:
                        all_sequences.append(sequence)
                        file_count += 1

                        if file_count % 50 == 0:
                            print(f"  Loaded {file_count} files...")

                        if max_files and file_count >= max_files:
                            break
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")

        if max_files and file_count >= max_files:
            break

    print(f"Loaded {file_count} files with {sum(len(s) for s in all_sequences):,} total tokens (including style prefixes)")
    return all_sequences


def concatenate_sequences(sequences, add_separator=True, separator_token=0):
    """
    Concatenate all sequences into a single sequence.

    Args:
        sequences: List of token sequences
        add_separator: Whether to add separator token between sequences
        separator_token: Token ID to use as separator (default: <PAD> = 0)

    Returns:
        Single concatenated sequence
    """
    if not sequences:
        return []

    result = []
    for i, seq in enumerate(sequences):
        result.extend(seq)
        if add_separator and i < len(sequences) - 1:
            result.append(separator_token)

    return result


def load_midi_dataset_with_style(
    data_dir='tokenized_midi_int',
    labels_path='midi_labels.json',
    vocab_path='vocab.json',
    train_ratio=0.9,
    add_separators=True,
    max_files=None,
    device='cpu'
):
    """
    Load complete MIDI dataset with style conditioning for training.

    Args:
        data_dir: Directory with tokenized integer files
        labels_path: Path to style labels JSON
        vocab_path: Path to vocabulary JSON
        train_ratio: Ratio of data to use for training (rest for validation)
        add_separators: Whether to add separator tokens between pieces
        max_files: Maximum files to load (None = all)
        device: Device to load tensors on

    Returns:
        Dictionary with:
            - train_data: Training data tensor
            - val_data: Validation data tensor
            - vocab: Vocabulary dictionary
            - vocab_size: Size of vocabulary
    """
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    print("="*60)
    print("LOADING MIDI DATASET WITH STYLE CONDITIONING")
    print("="*60)
    print(f"Vocabulary size: {vocab_size} tokens")

    # Load all sequences with style prefixes
    sequences = load_all_midi_data_with_style(data_dir, labels_path, vocab_path, max_files)

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

    print(f"\nDataset split:")
    print(f"  Training tokens: {len(train_data):,} ({100*len(train_data)/total_tokens:.1f}%)")
    print(f"  Validation tokens: {len(val_data):,} ({100*len(val_data)/total_tokens:.1f}%)")
    print("="*60)

    return {
        'train_data': train_data,
        'val_data': val_data,
        'vocab': vocab,
        'vocab_size': vocab_size,
    }


def get_batch(data, block_size, batch_size, device='cpu'):
    """
    Generate a batch of inputs and targets.

    Args:
        data: Full dataset tensor
        block_size: Context length
        batch_size: Number of sequences in batch
        device: Device to create tensors on

    Returns:
        x: Input batch (batch_size, block_size)
        y: Target batch (batch_size, block_size)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == '__main__':
    """Test the data loader with style conditioning."""

    print("Testing style-conditioned data loader...\n")

    # Load dataset
    dataset = load_midi_dataset_with_style(
        data_dir='tokenized_midi_int',
        labels_path='midi_labels.json',
        vocab_path='vocab.json',
        train_ratio=0.9,
        max_files=10  # Test with just 10 files
    )

    # Test batch generation
    block_size = 128
    batch_size = 4

    print("\nGenerating test batch...")
    xb, yb = get_batch(dataset['train_data'], block_size, batch_size)

    print(f"Input batch shape: {xb.shape}")
    print(f"Target batch shape: {yb.shape}")
    print(f"\nFirst sequence (first 30 tokens):")
    print(f"  Input:  {xb[0][:30].tolist()}")
    print(f"  Target: {yb[0][:30].tolist()}")

    # Decode to show style tokens
    reverse_vocab = {v: k for k, v in dataset['vocab'].items()}
    print(f"\nFirst 15 tokens decoded:")
    for i, token_id in enumerate(xb[0][:15].tolist()):
        token_name = reverse_vocab.get(token_id, f'UNK_{token_id}')
        print(f"  {i}: {token_id:4d} -> {token_name}")

    print("\n- Data loader test complete!")
