"""
MIDI Tokenization Pipeline

This script processes MIDI files and converts them into token sequences suitable
for machine learning models. It performs three main steps:

1. MIDI Tokenization: Reads MIDI files and converts them to text tokens:
   - NOTE_ON_{pitch} / NOTE_OFF_{pitch} for note events (pitch 0-127)
   - TIME_SHIFT_64ND_{n} for timing between events (in 64th note units)

2. Vocabulary Building: Creates a complete vocabulary mapping tokens to integers:
   - Special tokens: <PAD>, <START>, <END>
   - All possible note tokens (256 total: 128 NOTE_ON + 128 NOTE_OFF)
   - Time shift tokens up to a configurable maximum

3. Integer Conversion: Converts token text files to integer sequences for model input

Output files are organized under the dataset directory:
   - tokenized_midi/      - Text token files (*_tokens.txt)
   - tokenized_midi_int/  - Integer sequence files (*_int.txt)
   - vocab.json           - Token to integer mapping
   - reverse_vocab.json   - Integer to token mapping (for decoding)
"""

import os
import json
from collections import Counter

import mido
import pretty_midi
from typing import List
import glob
from tqdm import tqdm

datasets_dir = "./datasets"
dataset = "kaggle"

dataset_dir = f"{datasets_dir}/{dataset}"
dataset_input_dir = f"{dataset_dir}/data"
tokenized_output_dir = f"{dataset_dir}/tokenized_midi"
tokenized_ints_output_dir = f"{dataset_dir}/tokenized_midi_int"

# load and unpack midi file

def prepare_midi_file(input_file, output_file):
    mid = mido.MidiFile(input_file)

    # Print file info
    print(f"Type: {mid.type}")
    print(f"Ticks per beat: {mid.ticks_per_beat}")
    print(f"Number of tracks: {len(mid.tracks)}")
    print()

    # Print all messages
    for i, track in enumerate(mid.tracks):
        print(f"Track {i}: {track.name}")
        for msg in track:
            print(f"  {msg}")
        print()




TIME_RESOLUTION_MS = 10      # smallest time step
MAX_TIME_SHIFT_MS = 1000    # max single time-shift token
def ms_to_tokens(delta_ms: int, tempo: float = 120.0, division: int = 64) -> List[str]:
    """Convert a time delta (ms) into tokens representing time shifts.

    Args:
        delta_ms: Time delta in milliseconds
        tempo: Tempo in BPM
        division: Beat division (power of 2, e.g., 4=quarter, 8=eighth, 16=16th, 32=32nd)
    """
    # Calculate duration of one divided note in ms at given tempo
    # At 120 BPM: quarter note = 500ms
    ms_per_quarter = (60.0 / tempo) * 1000
    ms_per_division = ms_per_quarter / (division / 4)

    # Convert delta_ms to number of divided notes
    num_divisions = round(delta_ms / ms_per_division)

    tokens = []
    if num_divisions > 0:
        tokens.append(f"TIME_SHIFT_{division}ND_{num_divisions}")

    return tokens


def midi_to_tokens(midi_path: str) -> List[str]:

    midi = pretty_midi.PrettyMIDI(midi_path)

    events = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            events.append((note.start, "ON", note.pitch))
            events.append((note.end, "OFF", note.pitch))

    # Sort by time, NOTE_OFF before NOTE_ON at same time
    events.sort(key=lambda x: (x[0], x[1] == "ON"))

    tokens = []
    current_time = 0.0

    for time, kind, pitch in events:
        delta_ms = int((time - current_time) * 1000)

        if delta_ms > 0:
            tokens.extend(ms_to_tokens(delta_ms))
            current_time = time

        if kind == "ON":
            tokens.append(f"NOTE_ON_{pitch}")
        else:
            tokens.append(f"NOTE_OFF_{pitch}")

    return tokens


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


def convert_all_files(tokenized_dir, vocab, output_dir):
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
    note_on_count = sum(1 for k in vocab.keys() if k.startswith('NOTE_ON_'))
    note_off_count = sum(1 for k in vocab.keys() if k.startswith('NOTE_OFF_'))
    time_shift_count = sum(1 for k in vocab.keys() if k.startswith('TIME_SHIFT_'))
    special_count = sum(1 for k in vocab.keys() if k.startswith('<'))

    print(f"  Special tokens: {special_count}")
    print(f"  NOTE_ON tokens: {note_on_count} (covering MIDI notes 0-127)")
    print(f"  NOTE_OFF tokens: {note_off_count} (covering MIDI notes 0-127)")
    print(f"  TIME_SHIFT tokens: {time_shift_count}")

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


# Create output directory
os.makedirs(tokenized_output_dir, exist_ok=True)

# Find all MIDI files in kaggle_archive directory
midi_files = glob.glob(f"{dataset_input_dir}/**/*.mid", recursive=True)
midi_files.extend(glob.glob(f"{dataset_input_dir}/**/*.midi", recursive=True))

print(f"Found {len(midi_files)} MIDI files")

# Process each MIDI file with progress bar
for midi_file in tqdm(midi_files, desc="Tokenizing MIDI files"):
    midi_file_path = midi_file.replace("\\", "/")
    try:
        tokens = midi_to_tokens(midi_file_path)

        # Create output filename preserving directory structure
        rel_path = os.path.relpath(midi_file_path, f"{dataset_input_dir}")
        base_name = os.path.splitext(rel_path)[0]
        token_file = os.path.join(tokenized_output_dir, f"{base_name}_tokens.txt").replace("\\", "/")

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(token_file), exist_ok=True)

        # Save tokens to file
        with open(token_file, 'w') as f:
            for token in tokens:
                f.write(token + '\n')

    except Exception as e:
        print(f"\nError processing {midi_file}: {e}")

print(f"\nTokenization complete! Output saved to '{tokenized_output_dir}' directory")

# Build vocabulary from tokenized MIDI files
max_time_shift = 512  # Maximum time shift value in 64th notes

print("\nBuilding complete vocabulary for MIDI tokens...")
print(f"  - All MIDI notes: 0-127")
print(f"  - Time shifts: 0-{max_time_shift} (in 64th notes)")
print()

vocab, token_counts = build_vocabulary(tokenized_output_dir, max_time_shift)

# Save vocabulary
vocab_path = f"{dataset_dir}/vocab.json"
save_vocabulary(vocab, vocab_path)

# Print statistics
print_vocabulary_stats(vocab, token_counts)

# Convert all files to integer sequences
print("\nConverting tokenized files to integer sequences...")
convert_all_files(tokenized_output_dir, vocab, tokenized_ints_output_dir)

print("\nDone! Vocabulary created and all files converted.")
print(f"  - Vocabulary saved to: {vocab_path}")
print(f"  - Integer sequences saved to: {tokenized_ints_output_dir}/")

# Create reverse vocabulary for decoding
reverse_vocab = {v: k for k, v in vocab.items()}
reverse_vocab_path = f"{dataset_dir}/reverse_vocab.json"
with open(reverse_vocab_path, 'w') as f:
    json.dump(reverse_vocab, f, indent=2)
print(f"  - Reverse vocabulary saved to: {reverse_vocab_path}")
