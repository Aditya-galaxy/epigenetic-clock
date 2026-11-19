"""Command Line Interface for the Epigenetic Aging Clock Pipeline."""

import argparse
import sys
from src.pipeline import EpigeneticClockPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and benchmark human multi-tissue epigenetic aging clocks."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of donors in the synthetic multi-tissue cohort (default: 500).",
    )
    parser.add_argument(
        "--background-cpgs",
        type=int,
        default=250,
        help="Number of non-age-associated background CpGs (default: 250).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save figures, metrics, and models (default: results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    pipeline = EpigeneticClockPipeline(seed=args.seed)
    summary = pipeline.run_campaign(
        n_samples=args.samples,
        n_background_cpgs=args.background_cpgs,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
