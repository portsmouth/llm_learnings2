import os
import json
from collections import Counter
from pathlib import Path

def build_vocabulary(tokenized_dir, max_time_shift=512):
    """
    Build a complete vocabulary mapping tokens to integers.
    Generates tokens for ALL possible MIDI notes (0-127) and time shifts (0-max_time_shift).

    Args:
        tokenized_dir: Directory containing tokenized MIDI files (used to count token frequencies)
        max_time_shift: Maximum time shift value to include in vocabulary

    Returns:
        vocab: Dictionary mapping token strings to integer IDs
        token_counts: Counter showing frequency of each token
    """
    all_tokens = []

    # Walk through all tokenized files to count frequencies
    for root, dirs, files in os.walk(tokenized_dir):
        for file in files:
            if file.endswith('_tokens.txt'):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")

                with open(filepath, 'r') as f:
                    tokens = [line.strip() for line in f if line.strip()]
                    all_tokens.extend(tokens)

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Create complete vocabulary with ALL possible tokens
    # Special tokens: PAD, START, END
    special_tokens = ['<PAD>', '<START>', '<END>']

    # Generate ALL possible note tokens (0-127)
    note_on_tokens = [f'NOTE_ON_{i}' for i in range(128)]
    note_off_tokens = [f'NOTE_OFF_{i}' for i in range(128)]

    # Generate ALL possible time shift tokens
    time_shift_tokens = [f'TIME_SHIFT_64ND_{i}' for i in range(max_time_shift + 1)]

    # Build vocabulary mapping in order
    vocab = {}
    idx = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[token] = idx
        idx += 1

    # Add all NOTE_OFF tokens (0-127)
    for token in note_off_tokens:
        vocab[token] = idx
        idx += 1

    # Add all NOTE_ON tokens (0-127)
    for token in note_on_tokens:
        vocab[token] = idx
        idx += 1

    # Add all TIME_SHIFT tokens
    for token in time_shift_tokens:
        vocab[token] = idx
        idx += 1

    return vocab, token_counts

def save_vocabulary(vocab, output_path='vocab.json'):
    """Save vocabulary mapping to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {output_path}")
    print(f"Vocabulary size: {len(vocab)} tokens")

def tokenize_file(input_file, vocab, output_file):
    """
    Convert a tokenized text file to integer sequence.

    Args:
        input_file: Path to tokenized text file
        vocab: Vocabulary mapping
        output_file: Path to save integer sequence
    """
    with open(input_file, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]

    # Convert tokens to integers
    token_ids = [vocab[token] for token in tokens if token in vocab]

    # Save as space-separated integers
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, token_ids)))

    return token_ids

def convert_all_files(tokenized_dir, vocab, output_dir='tokenized_midi_int'):
    """
    Convert all tokenized files to integer sequences.

    Args:
        tokenized_dir: Directory with tokenized text files
        vocab: Vocabulary mapping
        output_dir: Directory to save integer sequences
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(tokenized_dir):
        for file in files:
            if file.endswith('_tokens.txt'):
                input_path = os.path.join(root, file)

                # Create corresponding output directory structure
                rel_path = os.path.relpath(root, tokenized_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Output filename: replace _tokens.txt with _int.txt
                output_filename = file.replace('_tokens.txt', '_int.txt')
                output_path = os.path.join(output_subdir, output_filename)

                # Convert file
                token_ids = tokenize_file(input_path, vocab, output_path)
                print(f"Converted {input_path} -> {output_path} ({len(token_ids)} tokens)")

def print_vocabulary_stats(vocab, token_counts):
    """Print statistics about the vocabulary."""
    print("\n" + "="*60)
    print("VOCABULARY STATISTICS")
    print("="*60)
    print(f"Total vocabulary size: {len(vocab)} tokens")

    # Count token types
    note_on_tokens = sum(1 for k in vocab.keys() if k.startswith('NOTE_ON_'))
    note_off_tokens = sum(1 for k in vocab.keys() if k.startswith('NOTE_OFF_'))
    time_shift_tokens = sum(1 for k in vocab.keys() if k.startswith('TIME_SHIFT_'))
    special_tokens = sum(1 for k in vocab.keys() if k.startswith('<'))

    print(f"  Special tokens: {special_tokens}")
    print(f"  NOTE_ON tokens: {note_on_tokens} (covering MIDI notes 0-127)")
    print(f"  NOTE_OFF tokens: {note_off_tokens} (covering MIDI notes 0-127)")
    print(f"  TIME_SHIFT tokens: {time_shift_tokens}")

    # Count how many tokens actually appear in the data
    tokens_in_data = sum(1 for token in vocab.keys() if token in token_counts)
    print(f"\nTokens actually used in dataset: {tokens_in_data}/{len(vocab)} ({100*tokens_in_data/len(vocab):.1f}%)")

    print("\nMost common tokens in dataset:")
    for token, count in token_counts.most_common(10):
        if token in vocab:
            print(f"  {token}: {count} occurrences (ID: {vocab[token]})")

    print("\nSample token mappings:")
    sample_tokens = ['<PAD>', '<START>', '<END>',
                     'NOTE_ON_0', 'NOTE_ON_60', 'NOTE_ON_127',
                     'NOTE_OFF_0', 'NOTE_OFF_60', 'NOTE_OFF_127',
                     'TIME_SHIFT_64ND_0', 'TIME_SHIFT_64ND_1', 'TIME_SHIFT_64ND_7']
    for token in sample_tokens:
        if token in vocab:
            count = token_counts.get(token, 0)
            usage = f" ({count} uses)" if count > 0 else " (unused)"
            print(f"  '{token}' -> {vocab[token]}{usage}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Build vocabulary from tokenized MIDI files
    tokenized_dir = 'tokenized_midi'
    max_time_shift = 512  # Maximum time shift value in 64th notes

    print("Building complete vocabulary for MIDI tokens...")
    print(f"  - All MIDI notes: 0-127")
    print(f"  - Time shifts: 0-{max_time_shift} (in 64th notes)")
    print()
    vocab, token_counts = build_vocabulary(tokenized_dir, max_time_shift)

    # Save vocabulary
    save_vocabulary(vocab, 'vocab.json')

    # Print statistics
    print_vocabulary_stats(vocab, token_counts)

    # Convert all files to integer sequences
    print("\nConverting tokenized files to integer sequences...")
    convert_all_files(tokenized_dir, vocab, 'tokenized_midi_int')

    print("\n✓ Done! Vocabulary created and all files converted.")
    print(f"  - Vocabulary saved to: vocab.json")
    print(f"  - Integer sequences saved to: tokenized_midi_int/")

    # Create reverse vocabulary for decoding
    reverse_vocab = {v: k for k, v in vocab.items()}
    with open('reverse_vocab.json', 'w') as f:
        json.dump(reverse_vocab, f, indent=2)
    print(f"  - Reverse vocabulary saved to: reverse_vocab.json")
