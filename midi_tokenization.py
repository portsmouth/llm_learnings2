

import mido
import os


import kagglehub

import kagglehub

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



import pretty_midi
from typing import List
import glob
from tqdm import tqdm

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


def velocity_to_bin(velocity: int, num_bins: int = 8) -> int:
    """
    Quantize MIDI velocity (0-127) into bins.

    Args:
        velocity: MIDI velocity value (0-127)
        num_bins: Number of bins to quantize into (default: 8)

    Returns:
        Bin number (0 to num_bins-1)
    """
    # Ensure velocity is in valid range
    velocity = max(0, min(127, velocity))
    # Quantize into bins
    bin_size = 128 / num_bins
    return min(int(velocity / bin_size), num_bins - 1)


def midi_to_tokens(midi_path: str, use_velocity: bool = True) -> List[str]:
    """
    Convert MIDI file to sequence of tokens.

    Args:
        midi_path: Path to MIDI file
        use_velocity: Whether to include velocity tokens (default: True)

    Returns:
        List of token strings
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    events = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            # Include velocity in the event tuple
            events.append((note.start, "ON", note.pitch, note.velocity))
            events.append((note.end, "OFF", note.pitch, 0))  # Note off doesn't need velocity

    # Sort by time, NOTE_OFF before NOTE_ON at same time
    events.sort(key=lambda x: (x[0], x[1] == "ON"))

    tokens = []
    current_time = 0.0

    for time, kind, pitch, velocity in events:
        delta_ms = int((time - current_time) * 1000)

        if delta_ms > 0:
            tokens.extend(ms_to_tokens(delta_ms))
            current_time = time

        if kind == "ON":
            if use_velocity:
                # Add velocity token before note on
                vel_bin = velocity_to_bin(velocity)
                tokens.append(f"VEL_{vel_bin}")
            tokens.append(f"NOTE_ON_{pitch}")
        else:
            tokens.append(f"NOTE_OFF_{pitch}")

    return tokens




# Create output directory
output_dir = "tokenized_midi"
os.makedirs(output_dir, exist_ok=True)

# Find all MIDI files in kaggle_archive directory
midi_files = glob.glob("kaggle_archive/**/*.mid", recursive=True)
midi_files.extend(glob.glob("kaggle_archive/**/*.midi", recursive=True))

print(f"Found {len(midi_files)} MIDI files")

# Process each MIDI file with progress bar
for midi_file in tqdm(midi_files, desc="Tokenizing MIDI files"):
    try:
        tokens = midi_to_tokens(midi_file)

        # Create output filename preserving directory structure
        rel_path = os.path.relpath(midi_file, "kaggle_archive")
        base_name = os.path.splitext(rel_path)[0]
        token_file = os.path.join(output_dir, f"{base_name}_tokens.txt")

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(token_file), exist_ok=True)

        # Save tokens to file
        with open(token_file, 'w') as f:
            for token in tokens:
                f.write(token + '\n')

    except Exception as e:
        print(f"\nError processing {midi_file}: {e}")

print(f"\nTokenization complete! Output saved to '{output_dir}' directory")


