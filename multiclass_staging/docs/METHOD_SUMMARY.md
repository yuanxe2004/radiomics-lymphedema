# Method summary

This project implements a leakage-controlled multiclass radiomics + morphology pipeline for lower-extremity lymphedema staging from CE-MRL-derived tabular features.

Core workflow:

1. Clean labels and normalize center identifiers.
2. Split data by center: internal centers for model development and one external center for independent testing.
3. Within the internal centers, perform group-aware train/validation/internal-test splitting.
4. Detect radiomics and morphology feature families.
5. Perform train-only feature selection inside each CV fold.
6. Compare candidate models with 5-fold cross-validation on train + validation data.
7. Select the global best model by AUC, then ACC, then PPV.
8. Refit the selected model on train + validation data.
9. Evaluate internal and external test sets with bootstrap 95% confidence intervals.
10. Save a reusable model wrapper for downstream prediction.

The original monolithic script used to derive this structure is preserved under `legacy/`.
