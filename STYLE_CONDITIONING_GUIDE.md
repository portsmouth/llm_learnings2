# Style-Conditioned MIDI Generation - Quick Start Guide

## Overview

Your MIDI generation model now supports **style conditioning**, allowing you to control the musical output by specifying:
- **Style/Period**: Baroque, Classical, Romantic, Impressionist
- **Tempo**: Very Slow, Slow, Moderate, Fast, Very Fast
- **Mood**: Bright, Dark, Dramatic, Calm, Playful
- **Dynamics**: Soft, Medium, Loud
- **Texture**: Sparse, Moderate, Rich
- **Form**: Sonata, Prelude, Waltz, Nocturne, etc.

## What Was Done

### 1. Automatic Labeling (✓ Complete)
All 295 MIDI files in your dataset have been automatically labeled with style tokens based on:
- **Filename analysis** (composer names, tempo keywords, musical forms)
- **MIDI metadata** (BPM, velocity, note density)

Results saved in `midi_labels.json`

### 2. Vocabulary Update (✓ Complete)
The vocabulary now includes 30+ style tokens:
- 4 style periods
- 5 tempo categories
- 5 mood types
- 3 dynamics levels
- 3 texture types
- 11 musical forms

**New vocabulary size: 811 tokens** (was ~780)

### 3. Style-Conditioned Data Loader (✓ Complete)
Created `load_midi_data_with_style.py` that:
- Loads MIDI sequences
- Prepends appropriate style tokens from `midi_labels.json`
- Ready for training

## How to Use

### Option 1: Train from Scratch with Style Conditioning

```bash
python train_with_style.py
```

This will:
- Load all MIDI data with style prefixes
- Train a transformer model to learn style-music relationships
- Save checkpoints as `ckpt_style_XXX.pt`
- Generate a sample with ROMANTIC + SLOW + CALM style

**Training time**: ~30-60 minutes on GPU for 5000 iterations

### Option 2: Generate Music with Style Prompts

After training, generate music with natural language prompts:

```bash
# Example 1: Romantic nocturne
python generate_with_style.py --prompt "slow romantic nocturne like Chopin"

# Example 2: Fast classical piece
python generate_with_style.py --prompt "fast bright classical Mozart sonata"

# Example 3: Dramatic piece
python generate_with_style.py --prompt "dramatic intense dark music"

# Example 4: Peaceful baroque
python generate_with_style.py --prompt "calm peaceful baroque prelude"
```

**Command-line options:**
```bash
python generate_with_style.py --help

Options:
  --prompt "text"         Style description
  --model PATH            Model checkpoint (default: midi_model_style.pth)
  --output PATH           Output MIDI file (default: generated_music.mid)
  --max-tokens N          Number of tokens to generate (default: 1000)
  --temperature N         Sampling temperature (default: 0.9)
  --top-k N               Top-k sampling parameter
  --seed N                Random seed for reproducibility
  --no-play               Don't play the generated MIDI
```

### Option 3: Use Existing Model Without Retraining

If you already have a trained model from `train_example.py`, you can:

1. **Continue training** with style conditioning:
   ```python
   # Modify train_with_style.py to load your existing checkpoint
   checkpoint = torch.load('midi_model.pth')
   model.load_state_dict(checkpoint['model'])
   # Then continue training with style-conditioned data
   ```

2. **Fine-tune** on a smaller learning rate with style data

## Style Token Examples

The prompt parser extracts style tokens from natural language:

| Prompt Keywords | Style Token |
|----------------|-------------|
| "baroque", "bach" | `<STYLE_BAROQUE>` |
| "classical", "mozart", "haydn" | `<STYLE_CLASSICAL>` |
| "romantic", "chopin", "liszt" | `<STYLE_ROMANTIC>` |
| "impressionist", "debussy", "ravel" | `<STYLE_IMPRESSIONIST>` |
| "very slow", "largo" | `<TEMPO_VERY_SLOW>` |
| "slow", "adagio", "andante" | `<TEMPO_SLOW>` |
| "moderate" | `<TEMPO_MODERATE>` |
| "fast", "allegro" | `<TEMPO_FAST>` |
| "very fast", "presto" | `<TEMPO_VERY_FAST>` |
| "bright", "happy", "cheerful" | `<MOOD_BRIGHT>` |
| "dark", "sad", "melancholic" | `<MOOD_DARK>` |
| "dramatic", "intense" | `<MOOD_DRAMATIC>` |
| "calm", "peaceful", "serene" | `<MOOD_CALM>` |
| "playful" | `<MOOD_PLAYFUL>` |
| "soft", "quiet" | `<DYN_SOFT>` |
| "loud" | `<DYN_LOUD>` |
| "sparse", "simple" | `<TEXTURE_SPARSE>` |
| "rich", "complex", "dense" | `<TEXTURE_RICH>` |

## Data Distribution (Your Dataset)

From the automatic labeling:

- **57.6%** Classical Era (Mozart, Beethoven, Haydn, Schubert)
- **38.3%** Romantic Era (Chopin, Schumann, Mendelssohn)
- **3.1%** Impressionist (Debussy, Ravel)
- **1.0%** Baroque (Bach)

**Tempo Distribution:**
- Very Slow: 11.5%
- Slow: 20.0%
- Moderate: 22.4%
- Fast: 28.1%
- Very Fast: 18.0%

## Files Overview

| File | Purpose |
|------|---------|
| `auto_label_midi.py` | Automatic labeling of MIDI files |
| `midi_labels.json` | Style labels for all MIDI files |
| `create_vocabulary.py` | Builds vocabulary with style tokens |
| `vocab.json` | Complete vocabulary (811 tokens) |
| `load_midi_data_with_style.py` | Data loader with style prefixes |
| `train_with_style.py` | Training script for style-conditioned model |
| `generate_with_style.py` | Generate music from text prompts |

## Advanced: Manual Style Token Selection

You can also specify style tokens directly in Python:

```python
from generate_with_style import load_vocab, load_model
import torch

# Load vocab and model
vocab = load_vocab('vocab.json')
model = load_model('midi_model_style.pth', len(vocab), device='cuda')

# Manually select style tokens
style_tokens = [
    '<STYLE_ROMANTIC>',
    '<TEMPO_SLOW>',
    '<MOOD_DARK>',
    '<DYN_SOFT>',
    '<TEXTURE_RICH>'
]

style_ids = [vocab[token] for token in style_tokens]
context = torch.tensor([style_ids], dtype=torch.long, device='cuda')

# Generate
generated = model.generate(context, max_new_tokens=1000, temperature=0.9)

# Convert to MIDI
from decode_midi import tokens_to_midi
tokens_to_midi(generated[0].cpu(), 'my_custom_music.mid')
```

## Tips for Better Generation

1. **Temperature**:
   - Lower (0.7-0.8) = more predictable, safer music
   - Higher (0.9-1.1) = more creative, riskier music

2. **Token Length**:
   - 500-1000 tokens = short piece (~30-60 seconds)
   - 1000-2000 tokens = medium piece (~1-2 minutes)

3. **Style Combinations**:
   - Use 3-5 style tokens for best results
   - Always include at least: STYLE, TEMPO, MOOD
   - Dynamics and texture are optional but helpful

4. **Prompt Clarity**:
   - More specific = better results
   - "slow romantic nocturne" > "nice music"
   - Include composer names for style hints

## Next Steps

1. **Train the model**: Run `python train_with_style.py`
2. **Test generation**: Try different style prompts
3. **Experiment**: Adjust temperature, token length, style combinations
4. **Fine-tune**: If results aren't good, train longer or adjust hyperparameters

## Troubleshooting

**Q: Model generates random music regardless of style?**
- The model needs sufficient training (3000+ iterations)
- Style conditioning is learned gradually
- Try more specific/consistent style prompts

**Q: Generation stops too early?**
- The model learned to emit `<END>` tokens
- Use `--max-tokens` to generate more
- Or retrain with fewer `<END>` tokens in data

**Q: MIDI playback sounds wrong?**
- Check that velocity information is present in tokens
- Verify tokenization includes `VEL_` tokens
- Try regenerating with different random seed

**Q: Out of memory during training?**
- Reduce `batch_size` (try 32 or 16)
- Reduce `block_size` (try 128)
- Use CPU instead of GPU (slower but works)
