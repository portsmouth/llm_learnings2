# Text-Conditioned MIDI Generation

## Overview

This document explains how to add text prompting to your MIDI generation model, allowing users to describe the desired music style in natural language.

## Architecture Options

### Option 1: Prefix Conditioning (Simplest)

**Implementation:**
1. Add style tokens to vocabulary (e.g., `<TEMPO_FAST>`, `<STYLE_CHOPIN>`)
2. Prepend style tokens to the beginning of sequences during training
3. At generation time, start with user-selected style tokens

**Example:**
```python
# Vocabulary additions
style_tokens = {
    # Tempo
    '<TEMPO_VERY_SLOW>': 'Largo/Grave (♩=40-60)',
    '<TEMPO_SLOW>': 'Adagio/Andante (♩=60-80)',
    '<TEMPO_MODERATE>': 'Moderato (♩=80-120)',
    '<TEMPO_FAST>': 'Allegro (♩=120-168)',
    '<TEMPO_VERY_FAST>': 'Presto (♩=168-200)',

    # Mood/Character
    '<MOOD_BRIGHT>': 'Happy, cheerful, energetic',
    '<MOOD_DARK>': 'Sad, melancholic, somber',
    '<MOOD_DRAMATIC>': 'Intense, passionate, emotional',
    '<MOOD_CALM>': 'Peaceful, serene, gentle',
    '<MOOD_PLAYFUL>': 'Light, whimsical, dance-like',

    # Style/Composer
    '<STYLE_BAROQUE>': 'Bach-like: contrapuntal, ornate',
    '<STYLE_CLASSICAL>': 'Mozart/Haydn: balanced, elegant',
    '<STYLE_ROMANTIC>': 'Chopin/Liszt: expressive, virtuosic',

    # Dynamics
    '<DYN_SOFT>': 'Piano (soft)',
    '<DYN_MEDIUM>': 'Mezzo-forte (medium)',
    '<DYN_LOUD>': 'Forte (loud)',

    # Texture
    '<TEXTURE_SPARSE>': 'Few notes, minimal accompaniment',
    '<TEXTURE_RICH>': 'Many notes, full chords',
}
```

**Training Data Preparation:**
```python
# Tag your MIDI files with style descriptors
training_examples = {
    'chopin_nocturne_op9_no2.mid': ['<TEMPO_SLOW>', '<MOOD_CALM>', '<STYLE_ROMANTIC>', '<DYN_SOFT>'],
    'mozart_sonata_k545.mid': ['<TEMPO_MODERATE>', '<MOOD_BRIGHT>', '<STYLE_CLASSICAL>'],
    'bach_prelude_bwv846.mid': ['<TEMPO_MODERATE>', '<MOOD_CALM>', '<STYLE_BAROQUE>'],
}

# Prepend to tokenized sequences
for midi_file, style_tags in training_examples.items():
    tokens = tokenize_midi(midi_file)
    conditioned_tokens = style_tags + tokens  # Prepend style tokens
```

**Generation:**
```python
# User selects style
style_tags = ['<TEMPO_FAST>', '<MOOD_DRAMATIC>', '<STYLE_ROMANTIC>']
style_ids = [vocab[tag] for tag in style_tags]

# Generate
context = torch.tensor([style_ids], device=device)
generated = model.generate(context, max_new_tokens=1000)
```

---

### Option 2: LLM-Powered Style Selection

**Implementation:**
1. Use an LLM to interpret natural language prompts
2. LLM maps user intent to style tokens
3. MIDI model generates based on selected tokens

**System Architecture:**
```
User Prompt → LLM (GPT-4/Claude) → Style Tokens → MIDI Model → Generated Music
```

**Example Code:**
```python
import anthropic

def prompt_to_style_tokens(user_prompt: str) -> List[str]:
    """
    Convert natural language prompt to style tokens using Claude.

    Args:
        user_prompt: User's description of desired music

    Returns:
        List of style token strings
    """
    client = anthropic.Anthropic(api_key="your-api-key")

    system_message = """You are a music style classifier. Convert natural language
descriptions of music into style tokens.

Available tokens:
- TEMPO: VERY_SLOW, SLOW, MODERATE, FAST, VERY_FAST
- MOOD: BRIGHT, DARK, DRAMATIC, CALM, PLAYFUL, MYSTERIOUS, HEROIC
- STYLE: BAROQUE, CLASSICAL, ROMANTIC, IMPRESSIONIST
- DYNAMICS: SOFT, MEDIUM, LOUD, VARIED
- TEXTURE: SPARSE, MODERATE, RICH

Output ONLY a comma-separated list of tokens (e.g., "TEMPO_FAST, MOOD_DRAMATIC, STYLE_ROMANTIC").
Do not include explanations."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"Convert this music request: {user_prompt}"
        }],
        system=system_message
    )

    # Parse response
    token_str = response.content[0].text.strip()
    tokens = [f"<{t.strip()}>" for t in token_str.split(',')]
    return tokens

# Usage
user_input = "Create a slow, romantic piano piece like a Chopin nocturne"
style_tokens = prompt_to_style_tokens(user_input)
# Returns: ['<TEMPO_SLOW>', '<MOOD_CALM>', '<STYLE_ROMANTIC>', '<DYN_SOFT>']

# Generate MIDI
style_ids = [vocab[token] for token in style_tokens]
context = torch.tensor([style_ids], device=device)
generated = model.generate(context, max_new_tokens=1000)
```

---

### Option 3: Cross-Attention Conditioning (Most Powerful)

**Implementation:**
Modify the transformer architecture to accept text embeddings via cross-attention.

**Architecture Changes:**
```python
class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention to text embeddings"""

    def __init__(self, block_size, n_embed, n_head=4, dropout=0.0):
        super().__init__()
        # Self-attention (as before)
        self.self_attn = MultiHeadAttention(...)
        self.ln1 = nn.LayerNorm(n_embed)

        # Cross-attention to text embeddings
        self.cross_attn = MultiHeadCrossAttention(n_embed, n_head, dropout)
        self.ln_cross = nn.LayerNorm(n_embed)

        # Feedforward
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, text_embeddings=None):
        # Self-attention
        x = x + self.self_attn(self.ln1(x))

        # Cross-attention to text (if provided)
        if text_embeddings is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_embeddings)

        # Feedforward
        x = x + self.ffwd(self.ln2(x))
        return x
```

**Text Encoder:**
```python
from sentence_transformers import SentenceTransformer

class TextConditionedMIDIModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, ...):
        super().__init__()

        # MIDI model components
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

        # Text encoder (frozen)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Project text embeddings to model dimension
        self.text_proj = nn.Linear(384, n_embed)  # MiniLM output is 384-d

        # Cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(block_size, n_embed, n_head, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, text_prompt=None, targets=None):
        # Encode MIDI tokens
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(idx.size(1), device=idx.device))
        x = tok_emb + pos_emb

        # Encode text (if provided)
        text_emb = None
        if text_prompt is not None:
            with torch.no_grad():
                text_emb = self.text_encoder.encode(text_prompt, convert_to_tensor=True)
            text_emb = self.text_proj(text_emb).unsqueeze(0)

        # Apply cross-attention blocks
        for block in self.blocks:
            x = block(x, text_emb)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None
```

**Training:**
```python
# Pair MIDI files with text descriptions
training_data = [
    ("chopin_nocturne.mid", "A slow, romantic nocturne with gentle arpeggios"),
    ("mozart_sonata.mid", "An upbeat classical sonata in C major"),
    ("bach_prelude.mid", "A baroque prelude with flowing counterpoint"),
]

# Training loop
for midi_file, text_description in training_data:
    tokens = tokenize_midi(midi_file)
    logits, loss = model(tokens, text_prompt=text_description, targets=tokens[1:])
    loss.backward()
    optimizer.step()
```

**Generation:**
```python
user_prompt = "Generate a melancholic, slow piano piece inspired by Chopin"
context = torch.tensor([[vocab['<START>']]], device=device)
generated = model.generate(context, text_prompt=user_prompt, max_new_tokens=1000)
```

---

## Recommended Implementation Path

### Phase 1: Start Simple (Week 1)
1. Add 20-30 style tokens to vocabulary
2. Manually tag 50-100 MIDI files with appropriate style tokens
3. Retrain model with style-prefixed sequences
4. Test generation with different style combinations

### Phase 2: Add LLM Interface (Week 2)
1. Create `prompt_to_style_tokens()` function using Claude/GPT-4
2. Build simple CLI/web interface for text prompts
3. Map LLM output to your style tokens
4. Validate that style tokens actually influence generation

### Phase 3: Advanced Conditioning (Optional)
1. Implement cross-attention architecture
2. Collect/create text descriptions for training data
3. Train with paired text-MIDI data
4. Fine-tune for better text-music alignment

---

## Example: Complete Pipeline

```python
# complete_pipeline.py

import torch
from anthropic import Anthropic

def generate_music_from_text(prompt: str, model, vocab):
    """
    Complete pipeline: Text prompt → Style tokens → MIDI generation

    Args:
        prompt: Natural language description of desired music
        model: Trained MIDI generation model
        vocab: Token vocabulary

    Returns:
        Path to generated MIDI file
    """
    # Step 1: LLM converts prompt to style tokens
    client = Anthropic(api_key="your-key")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Convert to style tokens: {prompt}

Available: TEMPO_SLOW/MODERATE/FAST, MOOD_BRIGHT/DARK/CALM/DRAMATIC,
STYLE_BAROQUE/CLASSICAL/ROMANTIC, DYN_SOFT/MEDIUM/LOUD

Output comma-separated tokens only."""
        }]
    )

    # Step 2: Parse style tokens
    token_names = [f"<{t.strip()}>" for t in response.content[0].text.split(',')]
    token_ids = [vocab[name] for name in token_names if name in vocab]

    # Step 3: Generate MIDI
    context = torch.tensor([token_ids], device='cuda')
    model.eval()
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=1000, temperature=0.9)

    # Step 4: Convert to MIDI file
    midi_path = tokens_to_midi(generated[0].cpu(), 'generated_output.mid')

    print(f"Generated: {midi_path}")
    print(f"Style: {token_names}")
    return midi_path

# Usage
prompt = "Create an energetic, virtuosic piano piece like a Liszt etude"
midi_file = generate_music_from_text(prompt, model, vocab)
```

---

## Training Data Annotation

To make this work, you'll need to annotate your training MIDI files. Here's a helper script:

```python
# annotate_midi_files.py

import os
import json
from pathlib import Path

def auto_annotate_from_filename(filename: str) -> List[str]:
    """Guess style tokens from filename/metadata"""
    tokens = []

    # Extract composer
    if 'bach' in filename.lower():
        tokens.append('<STYLE_BAROQUE>')
    elif any(c in filename.lower() for c in ['mozart', 'haydn', 'beethoven']):
        tokens.append('<STYLE_CLASSICAL>')
    elif any(c in filename.lower() for c in ['chopin', 'liszt', 'brahms']):
        tokens.append('<STYLE_ROMANTIC>')

    # Extract tempo hints from filename
    if any(t in filename.lower() for t in ['presto', 'allegro', 'vivace']):
        tokens.append('<TEMPO_FAST>')
    elif any(t in filename.lower() for t in ['adagio', 'largo', 'lento']):
        tokens.append('<TEMPO_SLOW>')
    else:
        tokens.append('<TEMPO_MODERATE>')

    # Extract mood from piece type
    if 'nocturne' in filename.lower():
        tokens.extend(['<MOOD_CALM>', '<DYN_SOFT>'])
    elif 'etude' in filename.lower():
        tokens.extend(['<MOOD_DRAMATIC>', '<DYN_LOUD>'])
    elif 'prelude' in filename.lower():
        tokens.append('<MOOD_CALM>')

    return tokens if tokens else ['<STYLE_CLASSICAL>', '<TEMPO_MODERATE>']

# Annotate all MIDI files
annotations = {}
for midi_file in Path('kaggle_archive').rglob('*.mid'):
    style_tokens = auto_annotate_from_filename(midi_file.name)
    annotations[str(midi_file)] = style_tokens

# Save annotations
with open('midi_annotations.json', 'w') as f:
    json.dump(annotations, f, indent=2)
```

---

## Next Steps

1. **Choose your approach**: Start with Option 1 (simplest) or Option 2 (most practical)
2. **Annotate data**: Tag 50-100 MIDIs with style tokens
3. **Update vocabulary**: Add style tokens to `create_vocabulary.py`
4. **Modify tokenization**: Prepend style tokens during data preparation
5. **Retrain model**: Train with style-conditioned sequences
6. **Test**: Generate music with different style combinations
7. **Add LLM interface**: Connect Claude/GPT-4 for natural language prompts

Would you like me to implement any of these approaches for your codebase?
