# MIDI Velocity Upgrade

## Overview

The tokenization has been enhanced to include **MIDI velocity** information, allowing the model to learn and generate dynamics (soft/loud playing).

## Changes Made

### 1. Tokenization ([midi_tokenization.py](midi_tokenization.py))

**Added:**
- `velocity_to_bin()`: Quantizes MIDI velocity (0-127) into 8 bins
- Modified `midi_to_tokens()` to emit velocity tokens before NOTE_ON events

**Velocity Bins:**
```
Bin 0: Velocity 0-15    (pppp - very very soft)
Bin 1: Velocity 16-31   (ppp - very soft)
Bin 2: Velocity 32-47   (pp - soft)
Bin 3: Velocity 48-63   (p - quiet)
Bin 4: Velocity 64-79   (mp/mf - medium)
Bin 5: Velocity 80-95   (f - loud)
Bin 6: Velocity 96-111  (ff - very loud)
Bin 7: Velocity 112-127 (fff - extremely loud)
```

**Token Sequence Example:**
```
Before: TIME_SHIFT_64ND_4 -> NOTE_ON_60 -> TIME_SHIFT_64ND_16 -> NOTE_OFF_60
After:  TIME_SHIFT_64ND_4 -> VEL_4 -> NOTE_ON_60 -> TIME_SHIFT_64ND_16 -> NOTE_OFF_60
```

### 2. Decoding ([decode_midi.py](decode_midi.py))

**Added:**
- `bin_to_velocity()`: Converts velocity bin back to MIDI velocity value
- Modified `tokens_to_midi()` to track current velocity and apply it to NOTE_ON events

**Default velocity**: 64 (medium/mezzo-forte) when no VEL_ token precedes a NOTE_ON

### 3. Vocabulary Creation ([create_vocabulary.py](create_vocabulary.py))

**Added:**
- 8 velocity tokens: `VEL_0`, `VEL_1`, ..., `VEL_7`
- Updated statistics to show velocity token counts

**New vocabulary size**: ~780 tokens (was ~772)
- 3 special tokens (<PAD>, <START>, <END>)
- 8 velocity tokens (VEL_0 through VEL_7)
- 256 note tokens (128 NOTE_ON + 128 NOTE_OFF)
- 513 time shift tokens (TIME_SHIFT_64ND_0 through TIME_SHIFT_64ND_512)

## How to Regenerate Data

Since you already have tokenized data, you'll need to re-tokenize with velocity information:

### Step 1: Re-tokenize MIDI files
```bash
python midi_tokenization.py
```

This will process all MIDI files in `kaggle_archive/` and create new token files in `tokenized_midi/` **with velocity information**.

### Step 2: Rebuild vocabulary
```bash
python create_vocabulary.py
```

This will:
- Build a new vocabulary including VEL_ tokens
- Convert all token files to integer sequences
- Save to `tokenized_midi_int/`
- Create `vocab.json` and `reverse_vocab.json`

### Step 3: Train new model
```bash
python train_example.py
```

The model will now learn:
- Which notes to play
- When to play them
- **How loud/soft to play them** (NEW!)

## Benefits

1. **Expressive Music**: Model can generate dynamics (crescendo, decrescendo, accents)
2. **Realistic Performance**: Piano pieces will sound more human-like
3. **Musical Phrasing**: Dynamics are essential for musical expression

## Backward Compatibility

- Old models (without velocity) will not work with new data
- Old tokenized data will need to be regenerated
- Generated MIDI from old models will use default velocity (64)

## Validation

To verify velocity tokens are working:

```python
# Check a generated MIDI file
from decode_midi import load_reverse_vocab
reverse_vocab = load_reverse_vocab()

# Sample token IDs from generation
# Should see VEL_X tokens before NOTE_ON_Y tokens
```

## Technical Details

**Velocity Quantization:**
- MIDI velocity: 0-127 (128 values)
- Quantized to: 8 bins
- Reduces vocabulary size while preserving dynamics
- Bin centers: 8, 24, 40, 56, 72, 88, 104, 120

**Token Order:**
```
VEL_{bin} must immediately precede NOTE_ON_{pitch}
```

**Model Impact:**
- Vocabulary increases from ~772 to ~780 tokens (+1% vocab size)
- Sequence length increases by ~30% (one VEL token per note)
- Model needs to learn velocity patterns (requires more training)
