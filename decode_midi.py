"""
Utility functions to convert generated token sequences back to MIDI files.
"""

import json
import os
import sys
import pygame
from mido import MidiFile, MidiTrack, Message, MetaMessage


def load_reverse_vocab(vocab_path='reverse_vocab.json'):
    """Load the reverse vocabulary (ID -> token string)."""
    with open(vocab_path, 'r') as f:
        reverse_vocab = json.load(f)
    # Convert string keys to integers
    return {int(k): v for k, v in reverse_vocab.items()}


def bin_to_velocity(vel_bin: int, num_bins: int = 8) -> int:
    """
    Convert velocity bin back to MIDI velocity value.

    Args:
        vel_bin: Velocity bin (0 to num_bins-1)
        num_bins: Number of bins (default: 8)

    Returns:
        MIDI velocity value (0-127)
    """
    # Use the middle of each bin
    bin_size = 128 / num_bins
    velocity = int((vel_bin + 0.5) * bin_size)
    return max(0, min(127, velocity))


def tokens_to_midi(token_ids, output_path='generated_output.mid', ticks_per_beat=480):
    """
    Convert a sequence of token IDs to a MIDI file.

    Args:
        token_ids: List or tensor of token IDs
        output_path: Path to save the MIDI file
        ticks_per_beat: MIDI ticks per beat (480 is standard)

    Returns:
        Path to the generated MIDI file
    """
    # Convert tensor to list if needed
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    # Load reverse vocabulary
    reverse_vocab = load_reverse_vocab()

    # Create MIDI file
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Add tempo (120 BPM = 500000 microseconds per beat)
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))

    # Track currently playing notes to ensure proper note offs
    active_notes = {}  # {note: time_activated}
    current_time = 0
    current_velocity = 64  # Default velocity (mezzo-forte)

    # Convert ticks: 64th notes to MIDI ticks
    # 1 beat = 4 quarter notes = 16 sixteenth notes = 64 sixty-fourth notes
    # So 1 sixty-fourth note = ticks_per_beat / 16
    ticks_per_64th = ticks_per_beat // 16

    print(f"Converting {len(token_ids)} tokens to MIDI...")

    for token_id in token_ids:
        if token_id not in reverse_vocab:
            continue

        token = reverse_vocab[token_id]

        # Skip special tokens (PAD, START, END)
        if token.startswith('<') and token.endswith('>'):
            continue

        if token.startswith('VEL_'):
            # Extract velocity bin and convert to MIDI velocity
            vel_bin = int(token.split('_')[-1])
            current_velocity = bin_to_velocity(vel_bin)

        elif token.startswith('NOTE_ON_'):
            # Extract note number
            note = int(token.split('_')[-1])
            # Add note on event with current velocity
            track.append(Message('note_on', note=note, velocity=current_velocity, time=current_time))
            active_notes[note] = 0
            current_time = 0

        elif token.startswith('NOTE_OFF_'):
            # Extract note number
            note = int(token.split('_')[-1])
            # Add note off event
            track.append(Message('note_off', note=note, velocity=64, time=current_time))
            if note in active_notes:
                del active_notes[note]
            current_time = 0

        elif token.startswith('TIME_SHIFT_64ND_'):
            # Extract time shift amount
            shift = int(token.split('_')[-1])
            # Convert to MIDI ticks
            current_time += shift * ticks_per_64th

    # Close any remaining active notes
    for note in list(active_notes.keys()):
        track.append(Message('note_off', note=note, velocity=64, time=current_time))
        current_time = 0

    # Save MIDI file
    mid.save(output_path)
    print(f"- MIDI file saved: {output_path}")

    return output_path


def play_midi(midi_path):
    """
    Play a MIDI file using the system's default MIDI player.

    Args:
        midi_path: Path to MIDI file to play
    """
    if not os.path.exists(midi_path):
        print(f"❌ Error: MIDI file not found: {midi_path}")
        return

    if sys.platform == 'win32':
        os.startfile(midi_path)
        print(f"- Playing MIDI with system default player: {midi_path}")
    elif sys.platform == 'darwin':  # macOS
        os.system(f'open "{midi_path}"')
        print(f"- Playing MIDI with system default player: {midi_path}")
    else:  # Linux and others
        os.system(f'xdg-open "{midi_path}"')
        print(f"- Playing MIDI with system default player: {midi_path}")


def play_with_pygame(midi_path, loop=False):
    """
    Play a MIDI file using pygame (pure Python solution).

    Args:
        midi_path: Path to MIDI file to play
        loop: Whether to loop playback continuously

    Returns:
        True if playback succeeded, False otherwise
    """
    try:
        import pygame
        import time
        from mido import MidiFile

        if not os.path.exists(midi_path):
            print(f"❌ Error: MIDI file not found: {midi_path}")
            return False

        # Initialize pygame mixer with better audio settings
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.music.load(midi_path)

        # Get MIDI duration for progress display
        try:
            mid = MidiFile(midi_path)
            duration = mid.length
        except:
            duration = 0

        # Play the file
        loop_count = -1 if loop else 0
        pygame.mixer.music.play(loop_count)

        print(f"🎵 Playing: {os.path.basename(midi_path)}")
        if duration > 0:
            print(f"   Duration: {duration:.1f}s")
        if loop:
            print("   Mode: Loop (Ctrl+C to stop)")
        print()

        # Wait for playback to finish with progress bar
        start_time = time.time()
        try:
            while pygame.mixer.music.get_busy():
                elapsed = time.time() - start_time

                if duration > 0 and not loop:
                    # Show progress bar
                    progress = min(elapsed / duration, 1.0)
                    bar_length = 40
                    filled = int(progress * bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"\r[{bar}] {elapsed:.1f}s / {duration:.1f}s", end='', flush=True)

                time.sleep(0.1)
        except KeyboardInterrupt:
            pygame.mixer.music.stop()
            print("\n\n⏹ Playback stopped")
            return True

        print("\n\n- Playback complete")
        pygame.mixer.quit()
        return True

    except ImportError:
        print("❌ pygame not installed. Install with: pip install pygame")
        return False
    except Exception as e:
        print(f"❌ Error playing MIDI: {e}")
        return False


if __name__ == '__main__':
    """
    Test the MIDI conversion functionality.
    """
    import torch

    print("Testing MIDI token decoding...")
    print()

    # Load vocabulary
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)

    # Create a simple test sequence
    # START -> NOTE_ON_60 (middle C) -> TIME_SHIFT -> NOTE_OFF_60 -> END
    test_tokens = [
        vocab.get('NOTE_ON_60', 0),
        vocab.get('TIME_SHIFT_64ND_16', 0),  # Hold for 16 64th notes
        vocab.get('NOTE_OFF_60', 0),
        vocab.get('NOTE_ON_64', 0),
        vocab.get('TIME_SHIFT_64ND_16', 0),
        vocab.get('NOTE_OFF_64', 0),
        vocab.get('<END>', 0)
    ]

    print(f"Test sequence: {test_tokens}")
    print()

    # Convert to MIDI
    output_file = 'test_output.mid'
    tokens_to_midi(test_tokens, output_file)

    print()
    print("Try playing with:")
    print(f"  python play_midi.py {output_file}")
