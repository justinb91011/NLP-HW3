#!/usr/bin/env python3
"""
Generates random sentences from a given smoothed trigram model.
"""
import argparse
import logging
from pathlib import Path
import torch

from probs import LanguageModel

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in an earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the trained model"
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help="Number of sentences to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Maximum length of the generated sentences (default is 20)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda', 'mps'],
        help="Device to use for PyTorch (cpu, cuda, or mps if you're on a Mac)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Specify the hardware device for computation
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ or you do not have an MPS-enabled device.")
            exit(1)
    torch.set_default_device(args.device)
    
    log.info("Loading the language model...")
    lm = LanguageModel.load(args.model, device=args.device)

    log.info(f"Generating {args.num_sentences} sentences with max length {args.max_length}...")
    sentences = lm.sample(k=args.num_sentences, max_length=args.max_length)

    # Print each generated sentence
    for sentence in sentences:
        print(' '.join(sentence))

if __name__ == "__main__":
    main()