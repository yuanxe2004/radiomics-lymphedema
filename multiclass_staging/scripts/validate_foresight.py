# -*- coding: utf-8 -*-
"""Prospective/foresight validation entry point for a saved multiclass wrapper."""

import argparse
from radiomics_lymphedema.validation import validate_table


def main():
    parser = argparse.ArgumentParser(description="Validate a saved multiclass model on a prospective Excel table.")
    parser.add_argument("--model", required=True, help="Path to best_model_wrapper.joblib")
    parser.add_argument("--input", required=True, help="Input Excel file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--label-col", default="label", help="True label column name in validation table")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations")
    args = parser.parse_args()
    validate_table(args.model, args.input, args.output_dir, args.label_col, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
