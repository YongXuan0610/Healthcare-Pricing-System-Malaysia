"""Person-calibrated quantile bounds for the practical 15-input model."""

import numpy as np
import pandas as pd


def bounds(artifact, frame, point=None):
    lower = artifact["lower"].predict(frame)
    upper = np.maximum(lower, artifact["upper"].predict(frame))
    lower = np.maximum(0, lower - artifact["adjustment"])
    upper = upper + artifact["adjustment"]
    if point is not None:
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("Non-finite practical 15 prediction interval")
    return lower, upper


def calibration_adjustment(lower, upper, target, people, coverage=0.90):
    scores = np.maximum.reduce(
        [lower - target, target - upper, np.zeros(len(target))]
    )
    person_scores = (
        pd.Series(scores)
        .groupby(np.asarray(people))
        .max()
        .sort_values()
        .to_numpy()
    )
    rank = int(np.ceil((len(person_scores) + 1) * coverage))
    if rank > len(person_scores):
        raise ValueError("Not enough calibration people for requested coverage")
    return float(person_scores[rank - 1]), rank, len(person_scores)
