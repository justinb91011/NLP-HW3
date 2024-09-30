#!/usr/bin/env python3
"""
Classifies text files as gen or spam using two language models and Bayes' Theorem
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gen_model",
        type=Path,
        help="path to the trained gen model",
    )
    parser.add_argument(
        "spam_model",
        type=Path,
        help="path to the trained spam model", 
    )
    parser.add_argument(
        "prior_gen",
        type=float,
        help="prior probability of the first category (gen)",
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

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob

def classify_file(file: Path, gen_lm: LanguageModel, spam_lm: LanguageModel, prior_gen: float) -> str:
  """Classify a file as gen or spam using two language models and Bayes' Theorem."""
  log_prob_gen = file_log_prob(file, gen_lm)
  log_prob_spam = file_log_prob(file, spam_lm)

  # Bayes' Theorem: add log prior probabilities 
  log_prior_gen = math.log(prior_gen)
  log_prior_spam = math.log(1 - prior_gen)

  # Posterior log-probabilities 
  log_posterior_gen = log_prob_gen + log_prior_gen
  log_posterior_spam = log_prob_spam + log_prior_spam

  # Classify based on the larger posterior log-probability
  return "gen" if log_posterior_gen > log_posterior_spam else "spam"


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    # Load the two models
    gen_lm = LanguageModel.load(args.gen_model, device=args.device)
    spam_lm = LanguageModel.load(args.spam_model, device=args.device)

    # Ensure both models have the same vocabulary 
    if gen_lm.vocab != spam_lm.vocab:
      raise ValueError("The vocabularies of the two models do not match. Please ensure both models are trained with the same vocabulary.")
    
    # Classify each file
    results = []
    for file in args.test_files:
        classification = classify_file(file, gen_lm, spam_lm, args.prior_gen)
        model_name = args.gen_model.name if classification == "gen" else args.spam_model.name
        print(f"{model_name}\t{file}")
        results.append(classification)

    # Calculate the number of files classified as gen and spam
    num_gen = results.count("gen")
    num_spam = results.count("spam")
    total_files = len(results)

    # Print summary with percentages
    perc_gen = (num_gen / total_files) * 100 if total_files > 0 else 0
    perc_spam = (num_spam / total_files) * 100 if total_files > 0 else 0

    print(f"\n{num_gen} files were more probably from {args.gen_model.name} ({perc_gen:.2f}%)")
    print(f"{num_spam} files were more probably from {args.spam_model.name} ({perc_spam:.2f}%)")


if __name__ == "__main__":
    main()

