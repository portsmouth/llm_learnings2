#!/usr/bin/env python3
"""
Command-line tool to play MIDI files using Python.
Uses pygame for pure Python playback without external dependencies.
"""

import os
import sys
import time
import argparse
import pygame
from mido import MidiFile


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

        print("\n\n✓ Playback complete")
        pygame.mixer.quit()
        return True

    except ImportError:
        print("❌ pygame not installed. Install with: pip install pygame")
        return False
    except Exception as e:
        print(f"❌ Error playing MIDI: {e}")
        return False


def play_with_system(midi_path):
    """
    Play a MIDI file using the system's default MIDI player.

    Args:
        midi_path: Path to MIDI file to play

    Returns:
        True if command executed, False otherwise
    """
    try:
        if not os.path.exists(midi_path):
            print(f"❌ Error: MIDI file not found: {midi_path}")
            return False

        print(f"🎵 Opening with system default player: {os.path.basename(midi_path)}")

        if sys.platform == 'win32':
            os.startfile(midi_path)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{midi_path}"')
        else:  # Linux and others
            os.system(f'xdg-open "{midi_path}"')

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_midi_info(midi_path):
    """
    Display information about a MIDI file.

    Args:
        midi_path: Path to MIDI file

    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(midi_path):
            print(f"❌ Error: MIDI file not found: {midi_path}")
            return False

        mid = MidiFile(midi_path)

        print(f"\n{'='*60}")
        print(f"MIDI File: {os.path.basename(midi_path)}")
        print(f"{'='*60}")
        print(f"Type: {mid.type}")
        print(f"Ticks per beat: {mid.ticks_per_beat}")
        print(f"Duration: {mid.length:.2f} seconds")
        print(f"Number of tracks: {len(mid.tracks)}")
        print()

        # Count events
        total_notes = 0
        total_messages = 0

        for i, track in enumerate(mid.tracks):
            note_count = sum(1 for msg in track if hasattr(msg, 'type') and msg.type in ['note_on', 'note_off'])
            message_count = len(track)
            total_notes += note_count
            total_messages += message_count

            print(f"Track {i}: {track.name if track.name else '(unnamed)'}")
            print(f"  Messages: {message_count}")
            print(f"  Notes: {note_count}")

        print()
        print(f"Total messages: {total_messages}")
        print(f"Total notes: {total_notes}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"❌ Error reading MIDI file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Play MIDI files from the command line using Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mid                    Play MIDI file with pygame
  %(prog)s --info song.mid             Show MIDI file information
  %(prog)s --loop song.mid             Loop playback continuously
  %(prog)s *.mid                       Play multiple files

Note: This tool uses pygame (pure Python) for playback.
Install pygame with: pip install pygame
        """
    )

    parser.add_argument('midi_files', nargs='+', metavar='MIDI_FILE',
                        help='MIDI file(s) to play')
    parser.add_argument('-s', '--system', action='store_true',
                        help='Use system default player instead of pygame')
    parser.add_argument('-i', '--info', action='store_true',
                        help='Show MIDI file information instead of playing')
    parser.add_argument('-l', '--loop', action='store_true',
                        help='Loop playback (pygame only)')

    args = parser.parse_args()

    # Process each MIDI file
    success_count = 0
    fail_count = 0

    for midi_file in args.midi_files:
        # Check if file exists
        if not os.path.isfile(midi_file):
            print(f"❌ File not found: {midi_file}")
            fail_count += 1
            continue

        # Check file extension
        if not midi_file.lower().endswith(('.mid', '.midi')):
            print(f"⚠ Warning: {midi_file} may not be a MIDI file")

        print()

        # Show info or play
        if args.info:
            success = show_midi_info(midi_file)
        elif args.system:
            success = play_with_system(midi_file)
        else:
            # Default to pygame (pure Python solution)
            success = play_with_pygame(midi_file, loop=args.loop)

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Add spacing between files if processing multiple
        if len(args.midi_files) > 1 and midi_file != args.midi_files[-1]:
            print("\n" + "-" * 60)

    # Summary
    if len(args.midi_files) > 1:
        print(f"\n{'='*60}")
        print(f"Processed {len(args.midi_files)} files: {success_count} succeeded, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
