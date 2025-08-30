
import math
import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _as_2d(df_row, cols):
    # ensure 2D dataframe keeping column order
    return df_row[cols].to_frame().T

def predict_tabular_prob(row_df, pipeline, expected_cols=None):
    """
    row_df: pandas.Series (a single row from the original dataframe)
    pipeline: sklearn-like model/pipeline loaded from pickle
    expected_cols: optional iterable of expected feature names; if None, tries pipeline.feature_names_in_
    Returns probability (float) of the positive class.
    """
    import pandas as pd
    if expected_cols is None and hasattr(pipeline, "feature_names_in_"):
        expected_cols = list(pipeline.feature_names_in_)
    elif expected_cols is None:
        expected_cols = list(row_df.index)

    X = _as_2d(row_df, [c for c in expected_cols if c in row_df.index])

    # Try predict_proba -> decision_function -> predict (as fallback 0/1)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        # assume binary: positive is column index 1, else max
        if proba.shape[1] == 2:
            return float(proba[0,1])
        else:
            return float(proba[0].max())
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)
        # scale via sigmoid (works for linear models)
        if np.ndim(score) == 2 and score.shape[1] == 1:
            score = score.ravel()
        return float(_sigmoid(score[0]))
    # last resort: predict -> cast to 0/1
    pred = pipeline.predict(X)
    return float(pred[0])
