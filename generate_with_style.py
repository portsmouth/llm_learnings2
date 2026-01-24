"""
Generate MIDI music from trained model with style prompting.
Allows specifying style, tempo, mood, dynamics, and texture.
"""

import torch
import json
import argparse
from models import SimpleTransformer
from decode_midi import tokens_to_midi, play_with_pygame


def load_vocab(vocab_path='vocab.json'):
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_model(model_path, vocab_size, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    # Check if this is a full checkpoint with config or just state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}

    # Get model parameters from config
    block_size = config.get('block_size', 256)
    n_embed = config.get('n_embed', 256)
    n_layer = config.get('n_layer', 3)
    n_head = config.get('n_head', 4)
    dropout = 0.0  # Always 0 during inference

    print(f"Loading model with: block_size={block_size}, n_embed={n_embed}, "
          f"n_layer={n_layer}, n_head={n_head}")

    model = SimpleTransformer(vocab_size, block_size, n_embed,
                             n_layer=n_layer, n_head=n_head, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_style_prompt(prompt: str, vocab: dict) -> list:
    """
    Parse a natural language style prompt into style tokens.

    Args:
        prompt: User's style description
        vocab: Vocabulary dictionary

    Returns:
        List of token IDs for the style
    """
    prompt_lower = prompt.lower()
    style_ids = []

    # Define keyword mappings
    style_keywords = {
        'baroque': '<STYLE_BAROQUE>',
        'bach': '<STYLE_BAROQUE>',
        'classical': '<STYLE_CLASSICAL>',
        'mozart': '<STYLE_CLASSICAL>',
        'haydn': '<STYLE_CLASSICAL>',
        'romantic': '<STYLE_ROMANTIC>',
        'chopin': '<STYLE_ROMANTIC>',
        'liszt': '<STYLE_ROMANTIC>',
        'impressionist': '<STYLE_IMPRESSIONIST>',
        'debussy': '<STYLE_IMPRESSIONIST>',
        'ravel': '<STYLE_IMPRESSIONIST>',
    }

    tempo_keywords = {
        'very slow': '<TEMPO_VERY_SLOW>',
        'slow': '<TEMPO_SLOW>',
        'moderate': '<TEMPO_MODERATE>',
        'fast': '<TEMPO_FAST>',
        'very fast': '<TEMPO_VERY_FAST>',
        'presto': '<TEMPO_VERY_FAST>',
        'allegro': '<TEMPO_FAST>',
        'andante': '<TEMPO_SLOW>',
        'adagio': '<TEMPO_SLOW>',
    }

    mood_keywords = {
        'bright': '<MOOD_BRIGHT>',
        'happy': '<MOOD_BRIGHT>',
        'cheerful': '<MOOD_BRIGHT>',
        'dark': '<MOOD_DARK>',
        'sad': '<MOOD_DARK>',
        'melancholic': '<MOOD_DARK>',
        'dramatic': '<MOOD_DRAMATIC>',
        'intense': '<MOOD_DRAMATIC>',
        'calm': '<MOOD_CALM>',
        'peaceful': '<MOOD_CALM>',
        'serene': '<MOOD_CALM>',
        'playful': '<MOOD_PLAYFUL>',
    }

    dynamics_keywords = {
        'soft': '<DYN_SOFT>',
        'quiet': '<DYN_SOFT>',
        'medium': '<DYN_MEDIUM>',
        'loud': '<DYN_LOUD>',
    }

    texture_keywords = {
        'sparse': '<TEXTURE_SPARSE>',
        'simple': '<TEXTURE_SPARSE>',
        'moderate': '<TEXTURE_MODERATE>',
        'rich': '<TEXTURE_RICH>',
        'complex': '<TEXTURE_RICH>',
        'dense': '<TEXTURE_RICH>',
    }

    # Check each keyword category
    for keyword, token in style_keywords.items():
        if keyword in prompt_lower:
            if token in vocab:
                style_ids.append(vocab[token])
            break

    for keyword, token in tempo_keywords.items():
        if keyword in prompt_lower:
            if token in vocab:
                style_ids.append(vocab[token])
            break

    for keyword, token in mood_keywords.items():
        if keyword in prompt_lower:
            if token in vocab:
                style_ids.append(vocab[token])
            break

    for keyword, token in dynamics_keywords.items():
        if keyword in prompt_lower:
            if token in vocab:
                style_ids.append(vocab[token])
            break

    for keyword, token in texture_keywords.items():
        if keyword in prompt_lower:
            if token in vocab:
                style_ids.append(vocab[token])
            break

    # Add neutral defaults for missing categories to match training distribution
    # During training, every sequence has exactly 5 style tokens (STYLE, TEMPO, MOOD, DYN, TEXTURE)
    # We need to provide all 5 during generation, using neutral defaults for unspecified ones

    # Check what's missing and add neutral defaults
    has_style = any(vocab.get(f'<STYLE_{s}>') in style_ids for s in ['BAROQUE', 'CLASSICAL', 'ROMANTIC', 'IMPRESSIONIST'])
    has_tempo = any(vocab.get(f'<TEMPO_{t}>') in style_ids for t in ['VERY_SLOW', 'SLOW', 'MODERATE', 'FAST', 'VERY_FAST'])
    has_mood = any(vocab.get(f'<MOOD_{m}>') in style_ids for m in ['BRIGHT', 'DARK', 'DRAMATIC', 'CALM', 'PLAYFUL'])
    has_dyn = any(vocab.get(f'<DYN_{d}>') in style_ids for d in ['SOFT', 'MEDIUM', 'LOUD'])
    has_texture = any(vocab.get(f'<TEXTURE_{t}>') in style_ids for t in ['SPARSE', 'MODERATE', 'RICH'])

    # Add neutral defaults for missing categories (most common in dataset)
    if not has_style and '<STYLE_CLASSICAL>' in vocab:
        style_ids.append(vocab['<STYLE_CLASSICAL>'])  # Most common
    if not has_tempo and '<TEMPO_MODERATE>' in vocab:
        style_ids.append(vocab['<TEMPO_MODERATE>'])  # Neutral
    if not has_mood and '<MOOD_CALM>' in vocab:
        style_ids.append(vocab['<MOOD_CALM>'])  # Neutral
    if not has_dyn and '<DYN_MEDIUM>' in vocab:
        style_ids.append(vocab['<DYN_MEDIUM>'])  # Neutral
    if not has_texture and '<TEXTURE_MODERATE>' in vocab:
        style_ids.append(vocab['<TEXTURE_MODERATE>'])  # Neutral

    return style_ids


def generate_from_prompt(
    prompt: str,
    model_path='midi_model_style.pth',
    vocab_path='vocab.json',
    output_path='generated_music.mid',
    max_tokens=1000,
    temperature=0.9,
    top_k=None,
    seed=None,
    device='cuda',
    play=True
):
    """
    Generate MIDI from a natural language prompt.

    Args:
        prompt: Style description (e.g., "fast romantic piece")
        model_path: Path to trained model
        vocab_path: Path to vocabulary
        output_path: Where to save generated MIDI
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
        device: Device to run on
        play: Whether to play the generated MIDI

    Returns:
        Path to generated MIDI file
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Load vocabulary
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    print("="*60)
    print("GENERATING MIDI FROM STYLE PROMPT")
    print("="*60)
    print(f"Prompt: \"{prompt}\"")
    print(f"Model: {model_path}")
    print(f"Vocabulary size: {vocab_size}")

    # Parse style prompt
    style_ids = parse_style_prompt(prompt, vocab)

    # Decode style tokens for display
    reverse_vocab = {v: k for k, v in vocab.items()}
    style_tokens = [reverse_vocab[id] for id in style_ids]

    print(f"\nStyle tokens extracted: {style_tokens}")
    print(f"Style token IDs: {style_ids}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, vocab_size, device)

    # Create context with style tokens
    if style_ids:
        context = torch.tensor([style_ids], dtype=torch.long, device=device)
    else:
        # Empty context if no style tokens
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    print(f"\nGenerating {max_tokens} tokens...")
    print(f"  Temperature: {temperature}")
    if top_k:
        print(f"  Top-k: {top_k}")

    end_token_id = vocab.get('<END>', None)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            end_token_id=end_token_id
        )

    print(f"Generated {generated.shape[1]} tokens total")

    # Convert to MIDI
    print(f"\nConverting to MIDI...")
    midi_path = tokens_to_midi(generated[0].cpu(), output_path)

    print("\n" + "="*60)
    print(f"- MIDI file saved: {midi_path}")
    print("="*60)

    # Play if requested
    if play:
        print("\nPlaying generated MIDI...")
        play_with_pygame(midi_path)

    return midi_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate MIDI music from style prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example prompts:
  "fast romantic piece like Chopin"
  "slow peaceful baroque music"
  "dramatic and intense classical sonata"
  "bright and cheerful Mozart-style piece"
  "dark melancholic nocturne"
        """
    )

    parser.add_argument('--prompt', type=str, default='romantic slow calm piece',
                       help='Style description for the music')
    parser.add_argument('--model', default='midi_model_style.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab', default='vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output', default='generated_music.mid',
                       help='Output MIDI file path')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-play', action='store_true',
                       help='Skip playing the generated MIDI')

    args = parser.parse_args()

    # Generate
    generate_from_prompt(
        prompt=args.prompt,
        model_path=args.model,
        vocab_path=args.vocab,
        output_path=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        play=not args.no_play
    )


if __name__ == '__main__':
    main()
