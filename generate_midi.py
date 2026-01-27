"""
Command-line tool to generate new MIDI files from trained models.

Usage:
    python generate_midi.py --model midi_model.pth --output my_music.mid --tokens 1000
    python generate_midi.py --model midi_model.pth --output my_music.mid --tokens 500 --temperature 1.2
    python generate_midi.py  # Uses defaults
"""

import argparse
import torch
import json
import os
import sys
from midiUtils import tokens_to_midi, play_midi
from models import BigramLanguageModel, SimpleTransformer
def load_vocab(vocab_path='vocab.json'):
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_model(model_path, vocab_size, device='cpu', model_type='auto', block_size=256, head_size=64, n_embed=256):
    """Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        vocab_size: Size of vocabulary
        device: Device to load model on
        model_type: 'bigram', 'transformer', or 'auto' to detect from checkpoint
        block_size: Context length (for transformer)
        head_size: Head size (for transformer)
        n_embed: Embedding dimension (for transformer)
    """
    # Load checkpoint to inspect architecture
    checkpoint = torch.load(model_path, map_location=device)

    # Auto-detect model type if needed
    if model_type == 'auto':
        if 'position_embedding_table.weight' in checkpoint:
            model_type = 'transformer'
        else:
            model_type = 'bigram'
        print(f"  Auto-detected model type: {model_type}")

    # Create appropriate model
    if model_type == 'transformer':
        # Infer parameters from checkpoint
        if 'position_embedding_table.weight' in checkpoint:
            block_size = checkpoint['position_embedding_table.weight'].shape[0]
        if 'token_embedding_table.weight' in checkpoint:
            n_embed = checkpoint['token_embedding_table.weight'].shape[1]
        # Head size is tricky - we can infer from sa_head structure
        if 'sa_head.heads.0.key.weight' in checkpoint:
            head_size = checkpoint['sa_head.heads.0.key.weight'].shape[0]
            # Multiply by number of heads to get total
            num_heads = sum(1 for k in checkpoint.keys() if k.startswith('sa_head.heads.') and k.endswith('.key.weight'))
            head_size = head_size * num_heads

        # Check if feedforward layer exists in checkpoint
        use_ffwd = 'ffwd.net.0.weight' in checkpoint

        print(f"  Inferred parameters: block_size={block_size}, n_embed={n_embed}, head_size={head_size}, use_ffwd={use_ffwd}")
        model = SimpleTransformer(vocab_size, block_size, head_size, n_embed, use_ffwd=use_ffwd).to(device)
    else:
        model = BigramLanguageModel(vocab_size).to(device)

    model.load_state_dict(checkpoint)
    model.eval()
    return model, model_type


def generate_midi_from_model(
    model_path='midi_model.pth',
    vocab_path='vocab.json',
    output_path='generated_music.mid',
    num_tokens=1000,
    temperature=1.0,
    top_k=None,
    seed=None,
    device=None,
    play=False
):
    """
    Generate a MIDI file from a trained model.

    Args:
        model_path: Path to the trained model (.pth file)
        vocab_path: Path to vocabulary JSON file
        output_path: Path where MIDI file will be saved
        num_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (default 1.0)
        top_k: If set, use top-k sampling
        seed: Random seed for reproducibility
        device: Device to use ('cuda' or 'cpu')
        play: Whether to play the MIDI after generation

    Returns:
        Path to generated MIDI file
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Check if vocab exists
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size}")

    # Load model
    print(f"Loading model from {model_path}...")
    model, model_type = load_model(model_path, vocab_size, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Prepare initial context
    end_token_id = vocab.get('<END>', None)
    start_token_id = vocab.get('<START>', None)

    if start_token_id is not None:
        context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
        print(f"  Using <START> token (ID={start_token_id}) as initial context")
    elif end_token_id is not None:
        context = torch.tensor([[end_token_id]], dtype=torch.long, device=device)
        print(f"  Using <END> token (ID={end_token_id}) as initial context")
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(f"  Using zero token as initial context")

    # Generate tokens
    print(f"\nGenerating up to {num_tokens} tokens...")
    print(f"  Temperature: {temperature}")
    if top_k:
        print(f"  Top-k: {top_k}")
    if end_token_id:
        print(f"  Will stop early if <END> token (ID={end_token_id}) is generated")

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            end_token_id=end_token_id
        )

    num_generated = generated.shape[1]
    print(f"✓ Generated {num_generated} tokens")

    # Convert to MIDI
    print(f"\nConverting to MIDI file: {output_path}")
    midi_path = tokens_to_midi(generated[0].cpu(), output_path)

    # Optionally play the MIDI
    if play:
        print("\nPlaying generated MIDI...")
        try:
            play_midi(midi_path)
        except Exception as e:
            print(f"Could not play MIDI: {e}")
            print("You can manually open the MIDI file in any MIDI player or DAW")

    print("\n" + "="*60)
    print(f"✓ Successfully generated: {midi_path}")
    print("="*60)

    return midi_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate MIDI files from trained language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default settings
  python generate_midi.py

  # Generate with custom model and output
  python generate_midi.py --model my_model.pth --output my_song.mid

  # Generate more tokens with higher randomness
  python generate_midi.py --tokens 2000 --temperature 1.5

  # Use top-k sampling for more focused output
  python generate_midi.py --top-k 50 --temperature 0.8

  # Generate and play immediately
  python generate_midi.py --play

  # Use specific random seed for reproducibility
  python generate_midi.py --seed 42
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='midi_model.pth',
        help='Path to trained model file (default: midi_model.pth)'
    )

    parser.add_argument(
        '--vocab', '-v',
        type=str,
        default='vocab.json',
        help='Path to vocabulary JSON file (default: vocab.json)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='generated_music.mid',
        help='Output MIDI file path (default: generated_music.mid)'
    )

    parser.add_argument(
        '--tokens', '-t',
        type=int,
        default=1000,
        help='Maximum number of tokens to generate (default: 1000)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (higher=more random, lower=more conservative, default: 1.0)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Use top-k sampling with k most likely tokens (optional)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use for generation (default: auto-detect)'
    )

    parser.add_argument(
        '--play', '-p',
        action='store_true',
        help='Play the generated MIDI file after creation'
    )

    args = parser.parse_args()

    # Generate MIDI
    generate_midi_from_model(
        model_path=args.model,
        vocab_path=args.vocab,
        output_path=args.output,
        num_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        play=args.play
    )


if __name__ == '__main__':
    main()
