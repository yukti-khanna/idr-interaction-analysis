#!/usr/bin/env python3
"""Plot feature-vs-B22 scatter plots from the merged analysis table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FEATURES = [
    "Opposite_Charge_Number",
    "Normalized_OCN",
    "Charge_Complementarity",
    "Average_lambda",
]


class PlotError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected features against B22_corr1_log10.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Merged features+B22 CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output plots.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to plot against B22_corr1_log10.",
    )
    parser.add_argument(
        "--y-column",
        default="B22_corr1_log10",
        help="Y-axis column. Default: B22_corr1_log10.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.input_csv.exists():
            raise PlotError(f"Input CSV not found: {args.input_csv}")

        df = pd.read_csv(args.input_csv)
        if args.y_column not in df.columns:
            raise PlotError(f"Y-axis column not found: {args.y_column}")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for feature in args.features:
            if feature not in df.columns:
                raise PlotError(f"Requested feature column not found: {feature}")

            plot_df = df[[feature, args.y_column]].dropna()
            plt.figure(figsize=(4.2, 3.6))
            plt.scatter(plot_df[feature], plot_df[args.y_column], s=18)
            plt.xlabel(feature)
            plt.ylabel(args.y_column)
            plt.tight_layout()

            png_path = args.output_dir / f"{feature}_vs_{args.y_column}.png"
            pdf_path = args.output_dir / f"{feature}_vs_{args.y_column}.pdf"
            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path)
            plt.close()

        print(f"Wrote plots to {args.output_dir}")
        return 0
    except PlotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
