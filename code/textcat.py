#!/usr/bin/env python3
"""
Classifies text files using two language models and Bayes' Theorem
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "first_model",
        type=Path,
        help="path to the first model (e.g., en.model or gen.model)",
    )
    parser.add_argument(
        "second_model",
        type=Path,
        help="path to the second model (e.g., es.model or spam.model)", 
    )
    parser.add_argument(
        "prior_first",
        type=float,
        help="prior probability of the first category (e.g., en or gen)",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        if log_prob == -math.inf:
            break  # Stop early if log_prob is -infinity due to unseen n-grams

    return log_prob

def classify_file(file: Path, first_lm: LanguageModel, second_lm: LanguageModel, prior_first: float) -> str:
    """Classify a file using two language models and Bayes' Theorem."""
    log_prob_first = file_log_prob(file, first_lm)
    log_prob_second = file_log_prob(file, second_lm)

    # Bayes' Theorem: add log prior probabilities 
    log_prior_first = math.log(prior_first)
    log_prior_second = math.log(1 - prior_first)

    # Posterior log-probabilities 
    log_posterior_first = log_prob_first + log_prior_first
    log_posterior_second = log_prob_second + log_prior_second

    # Classify based on the larger posterior log-probability
    return "first" if log_posterior_first > log_posterior_second else "second"

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    torch.set_default_device(args.device)
    
    log.info("Testing...")
    
    # Load the two models
    first_lm = LanguageModel.load(args.first_model, device=args.device)
    second_lm = LanguageModel.load(args.second_model, device=args.device)

    # Ensure both models have the same vocabulary 
    if first_lm.vocab != second_lm.vocab:
        raise ValueError("The vocabularies of the two models do not match. Please ensure both models are trained with the same vocabulary.")
    
    # Classify each file and count results
    num_first = 0
    num_second = 0
    total_files = 0
    
    for file in args.test_files:
        classification = classify_file(file, first_lm, second_lm, args.prior_first)

        # Determine the name of the model based on classification
        model_name = args.first_model.name if classification == "first" else args.second_model.name
        print(f"{model_name}\t{file}")

        # Count classifications for each model
        if classification == "first":
            num_first += 1
        else:
            num_second += 1

        total_files += 1

    # Calculate percentages
    perc_first = (num_first / total_files) * 100 if total_files > 0 else 0
    perc_second = (num_second / total_files) * 100 if total_files > 0 else 0

    # Keep print statements for gen/spam unchanged but generalized for model names
    print(f"\n{num_first} files were more probably from {args.first_model.name} ({perc_first:.2f}%)")
    print(f"{num_second} files were more probably from {args.second_model.name} ({perc_second:.2f}%)")


if __name__ == "__main__":
    main()