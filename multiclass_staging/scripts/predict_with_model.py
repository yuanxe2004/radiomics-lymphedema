# -*- coding: utf-8 -*-
"""Predict labels for a new Excel table using a saved best_model_wrapper.joblib."""

import argparse
import pandas as pd
from radiomics_lymphedema.wrapper import load_best_model


def main():
    parser = argparse.ArgumentParser(description="Run multiclass prediction with a saved model wrapper.")
    parser.add_argument("--model", required=True, help="Path to best_model_wrapper.joblib")
    parser.add_argument("--input", required=True, help="Input Excel file with required feature columns")
    parser.add_argument("--output", required=True, help="Output Excel file")
    args = parser.parse_args()

    wrapper = load_best_model(args.model)
    df = pd.read_excel(args.input)
    result = wrapper.predict_dataframe(df)
    result.to_excel(args.output, index=False)
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()
