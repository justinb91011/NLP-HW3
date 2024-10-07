#!/usr/bin/env python3
import argparse
import logging
import math
from pathlib import Path
import torch
from probs import Wordtype, LanguageModel, read_transcription_candidates, read_trigrams_from_sentence

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the trained model")
    parser.add_argument("test_files", type=Path, nargs="*")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda', 'mps'])
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING)
    return parser.parse_args()

def compute_log_posterior(lm: LanguageModel, likelihood: float, transcription: str) -> float:
    """Compute the log-posterior for a candidate transcription."""
    # Compute log p(u | transcription) + log p(transcription)
    log_prior = 0.0
    for (x, y, z) in read_trigrams_from_sentence(transcription.split(), lm.vocab):
        log_prior += lm.log_prob(x, y, z)
    
    log_posterior = likelihood + log_prior
    return log_posterior

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    torch.set_default_device(args.device)
    
    lm = LanguageModel.load(args.model, device=args.device)
    
    overall_error_count = 0
    overall_word_count = 0
    
    for test_file in args.test_files:
        best_log_posterior = -math.inf
        best_transcription = None
        correct_transcription = None
        likelihoods = []
        transcriptions = []
        correct_words = []

        # Read candidate transcriptions and correct transcription
        candidates = read_transcription_candidates(test_file)

        for candidate in candidates:
            likelihood, transcription, correct = candidate
            likelihoods.append(likelihood)
            transcriptions.append(transcription)
            if correct:
                correct_transcription = transcription

            # Compute posterior for each transcription
            log_posterior = compute_log_posterior(lm, likelihood, transcription)
            if log_posterior > best_log_posterior:
                best_log_posterior = log_posterior
                best_transcription = transcription

        # Handle cases where correct transcription is not found or correct_words is empty
        if not correct_transcription:
            print(f"No correct transcription found for {test_file}, skipping.")
            continue

        # Now, use correct_transcription (which is a string) and split it
        correct_words = correct_transcription.split()

        # Compute word error rate (WER)
        best_words = best_transcription.split()
        word_errors = sum(1 for b, c in zip(best_words, correct_words) if b != c) + abs(len(best_words) - len(correct_words))
        overall_error_count += word_errors
        overall_word_count += len(correct_words)

        error_rate = word_errors / len(correct_words) if len(correct_words) > 0 else 0
        print(f"{error_rate:.3f} {test_file}")
    
    if overall_word_count > 0:
        overall_error_rate = overall_error_count / overall_word_count
        print(f"{overall_error_rate:.3f} OVERALL")
    else:
        print("No valid correct transcriptions found in the input files.")


if __name__ == "__main__":
    main()