"""
Automatically label MIDI files based on filenames and metadata.
"""

import os
import json
import pretty_midi
from pathlib import Path
from collections import defaultdict


# Define style token mappings
COMPOSER_STYLES = {
    'bach': '<STYLE_BAROQUE>',
    'handel': '<STYLE_BAROQUE>',
    'vivaldi': '<STYLE_BAROQUE>',
    'scarlatti': '<STYLE_BAROQUE>',

    'mozart': '<STYLE_CLASSICAL>',
    'haydn': '<STYLE_CLASSICAL>',
    'beethoven': '<STYLE_CLASSICAL>',
    'clementi': '<STYLE_CLASSICAL>',

    'chopin': '<STYLE_ROMANTIC>',
    'liszt': '<STYLE_ROMANTIC>',
    'schumann': '<STYLE_ROMANTIC>',
    'brahms': '<STYLE_ROMANTIC>',
    'mendelssohn': '<STYLE_ROMANTIC>',
    'rachmaninoff': '<STYLE_ROMANTIC>',
    'tchaikovsky': '<STYLE_ROMANTIC>',

    'debussy': '<STYLE_IMPRESSIONIST>',
    'ravel': '<STYLE_IMPRESSIONIST>',
    'satie': '<STYLE_IMPRESSIONIST>',
}

TEMPO_KEYWORDS = {
    # Very slow
    'grave': '<TEMPO_VERY_SLOW>',
    'largo': '<TEMPO_VERY_SLOW>',
    'larghetto': '<TEMPO_VERY_SLOW>',

    # Slow
    'adagio': '<TEMPO_SLOW>',
    'lento': '<TEMPO_SLOW>',
    'andante': '<TEMPO_SLOW>',

    # Moderate
    'moderato': '<TEMPO_MODERATE>',
    'andantino': '<TEMPO_MODERATE>',
    'allegretto': '<TEMPO_MODERATE>',

    # Fast
    'allegro': '<TEMPO_FAST>',
    'vivace': '<TEMPO_FAST>',
    'allegro vivace': '<TEMPO_FAST>',

    # Very fast
    'presto': '<TEMPO_VERY_FAST>',
    'prestissimo': '<TEMPO_VERY_FAST>',
}

MOOD_KEYWORDS = {
    # Bright/Happy
    'allegro': '<MOOD_BRIGHT>',
    'vivace': '<MOOD_BRIGHT>',
    'giocoso': '<MOOD_BRIGHT>',
    'scherzando': '<MOOD_PLAYFUL>',
    'scherzo': '<MOOD_PLAYFUL>',

    # Dark/Sad
    'nocturne': '<MOOD_DARK>',
    'elegy': '<MOOD_DARK>',
    'funeral': '<MOOD_DARK>',
    'lament': '<MOOD_DARK>',

    # Calm
    'adagio': '<MOOD_CALM>',
    'largo': '<MOOD_CALM>',
    'pastorale': '<MOOD_CALM>',
    'berceuse': '<MOOD_CALM>',

    # Dramatic
    'etude': '<MOOD_DRAMATIC>',
    'fantasia': '<MOOD_DRAMATIC>',
    'ballade': '<MOOD_DRAMATIC>',
    'rhapsody': '<MOOD_DRAMATIC>',
}

FORM_KEYWORDS = {
    'sonata': '<FORM_SONATA>',
    'prelude': '<FORM_PRELUDE>',
    'fugue': '<FORM_FUGUE>',
    'waltz': '<FORM_WALTZ>',
    'mazurka': '<FORM_MAZURKA>',
    'nocturne': '<FORM_NOCTURNE>',
    'etude': '<FORM_ETUDE>',
    'impromptu': '<FORM_IMPROMPTU>',
    'scherzo': '<FORM_SCHERZO>',
    'polonaise': '<FORM_POLONAISE>',
    'concerto': '<FORM_CONCERTO>',
}


def analyze_midi_metadata(midi_path: str):
    """
    Extract metadata from MIDI file itself (tempo, note range, etc.)

    Returns:
        dict with metadata
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Get tempo (average BPM)
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            avg_tempo = tempo_changes[1].mean()
        else:
            avg_tempo = 120  # default

        # Get note statistics
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)

        if all_notes:
            velocities = [n.velocity for n in all_notes]
            avg_velocity = sum(velocities) / len(velocities)

            pitches = [n.pitch for n in all_notes]
            pitch_range = max(pitches) - min(pitches)

            # Note density (notes per second)
            duration = midi.get_end_time()
            note_density = len(all_notes) / duration if duration > 0 else 0
        else:
            avg_velocity = 64
            pitch_range = 0
            note_density = 0

        return {
            'tempo': avg_tempo,
            'avg_velocity': avg_velocity,
            'pitch_range': pitch_range,
            'note_density': note_density,
            'duration': midi.get_end_time(),
        }
    except Exception as e:
        print(f"Warning: Could not analyze {midi_path}: {e}")
        return None


def tempo_to_token(bpm: float) -> str:
    """Convert BPM to tempo token"""
    if bpm < 60:
        return '<TEMPO_VERY_SLOW>'
    elif bpm < 90:
        return '<TEMPO_SLOW>'
    elif bpm < 120:
        return '<TEMPO_MODERATE>'
    elif bpm < 150:
        return '<TEMPO_FAST>'
    else:
        return '<TEMPO_VERY_FAST>'


def velocity_to_dynamic_token(velocity: float) -> str:
    """Convert average velocity to dynamics token"""
    # Adjusted thresholds for classical piano dataset (tends to be soft)
    if velocity < 50:
        return '<DYN_SOFT>'
    elif velocity < 65:
        return '<DYN_MEDIUM>'
    else:
        return '<DYN_LOUD>'


def density_to_texture_token(density: float) -> str:
    """Convert note density to texture token"""
    if density < 5:
        return '<TEXTURE_SPARSE>'
    elif density < 15:
        return '<TEXTURE_MODERATE>'
    else:
        return '<TEXTURE_RICH>'


def label_from_filename(filepath: str) -> list:
    """
    Extract style tokens from filename.

    Args:
        filepath: Path to MIDI file

    Returns:
        List of style tokens
    """
    filename = os.path.basename(filepath).lower()
    filepath_lower = filepath.lower()

    tokens = []

    # Extract composer/style from filename or path
    for composer, style_token in COMPOSER_STYLES.items():
        if composer in filename or composer in filepath_lower:
            tokens.append(style_token)
            break  # Only one style

    # Extract tempo from filename
    for keyword, tempo_token in TEMPO_KEYWORDS.items():
        if keyword in filename:
            tokens.append(tempo_token)
            break  # Only one tempo

    # Extract mood from filename
    for keyword, mood_token in MOOD_KEYWORDS.items():
        if keyword in filename:
            tokens.append(mood_token)
            break  # Could have multiple moods, but keep it simple

    # Extract form from filename
    for keyword, form_token in FORM_KEYWORDS.items():
        if keyword in filename:
            tokens.append(form_token)
            break

    return tokens


def label_from_metadata(midi_path: str) -> list:
    """
    Extract style tokens from MIDI metadata (tempo, velocity, etc.)

    Args:
        midi_path: Path to MIDI file

    Returns:
        List of style tokens
    """
    metadata = analyze_midi_metadata(midi_path)
    if metadata is None:
        return []

    tokens = []

    # Add tempo token based on BPM
    tokens.append(tempo_to_token(metadata['tempo']))

    # Add dynamics token based on average velocity
    tokens.append(velocity_to_dynamic_token(metadata['avg_velocity']))

    # Add texture token based on note density
    tokens.append(density_to_texture_token(metadata['note_density']))

    return tokens


def auto_label_midi_files(midi_dir: str, output_file: str = 'midi_labels.json',
                          use_metadata: bool = True):
    """
    Automatically label all MIDI files in directory.

    Args:
        midi_dir: Directory containing MIDI files
        output_file: Where to save labels JSON
        use_metadata: Whether to analyze MIDI metadata (slower but more accurate)
    """
    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(Path(midi_dir).rglob(ext))

    print(f"Found {len(midi_files)} MIDI files")

    labels = {}
    stats = defaultdict(int)

    for i, midi_path in enumerate(midi_files):
        if i % 50 == 0:
            print(f"Processing {i}/{len(midi_files)}...")

        # Get tokens from filename
        filename_tokens = label_from_filename(str(midi_path))

        # Get tokens from metadata (optional)
        if use_metadata:
            metadata_tokens = label_from_metadata(str(midi_path))
        else:
            metadata_tokens = []

        # Combine tokens (filename tokens override metadata)
        all_tokens = list(set(filename_tokens + metadata_tokens))

        # Ensure we have at least some defaults
        if not any(t.startswith('<STYLE_') for t in all_tokens):
            all_tokens.append('<STYLE_CLASSICAL>')  # Default style

        if not any(t.startswith('<TEMPO_') for t in all_tokens):
            all_tokens.append('<TEMPO_MODERATE>')  # Default tempo

        if not any(t.startswith('<MOOD_') for t in all_tokens):
            all_tokens.append('<MOOD_CALM>')  # Default mood

        # Store labels
        relative_path = str(midi_path.relative_to(midi_dir))
        labels[relative_path] = sorted(all_tokens)

        # Update statistics
        for token in all_tokens:
            stats[token] += 1

    # Save labels
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\n- Labeled {len(labels)} MIDI files")
    print(f"- Saved to {output_file}")

    # Print statistics
    print("\nLabel Distribution:")
    for token, count in sorted(stats.items()):
        percentage = 100 * count / len(labels)
        print(f"  {token}: {count} files ({percentage:.1f}%)")

    return labels


def review_labels(labels_file: str = 'midi_labels.json', num_samples: int = 10):
    """
    Review a sample of labeled files for quality checking.

    Args:
        labels_file: Path to labels JSON
        num_samples: Number of samples to show
    """
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    import random
    samples = random.sample(list(labels.items()), min(num_samples, len(labels)))

    print(f"\nSample of {len(samples)} labeled files:")
    print("=" * 80)
    for filepath, tokens in samples:
        filename = os.path.basename(filepath)
        print(f"\nFile: {filename}")
        print(f"Path: {filepath}")
        print(f"Labels: {', '.join(tokens)}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Auto-label MIDI files with style tokens')
    parser.add_argument('--midi-dir', default='kaggle_archive',
                       help='Directory containing MIDI files')
    parser.add_argument('--output', default='midi_labels.json',
                       help='Output file for labels')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip metadata analysis (faster but less accurate)')
    parser.add_argument('--review', action='store_true',
                       help='Review labeled samples')

    args = parser.parse_args()

    # Label files
    labels = auto_label_midi_files(
        args.midi_dir,
        args.output,
        use_metadata=not args.no_metadata
    )

    # Review samples
    if args.review:
        review_labels(args.output)
