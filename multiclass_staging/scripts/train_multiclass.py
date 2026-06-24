# -*- coding: utf-8 -*-
"""CLI entry for multiclass model training."""

import argparse
from radiomics_lymphedema.config import load_config
from radiomics_lymphedema.pipeline import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train multiclass LEL staging model.")
    parser.add_argument("--config", default=None, help="Path to YAML config. If omitted, package defaults are used.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
