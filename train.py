"""Train one model per stat and report out-of-time accuracy.

Split (by season start year):
  train  <= 2023 (2004-05 .. 2023-24)
  valid  =  2024 (2024-25)        -> picks the number of boosting rounds
  test   >= 2025 (2025-26 + 2026-27 to date) -> reported accuracy, interval calibration
Then refits on everything and saves to artifacts/model.joblib.

Usage: python train.py [--rebuild-data]
Takes ~30 min on a laptop (keep it awake: `caffeinate -i python train.py`).
"""
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nba_model.data import ARTIFACT_DIR, prepare
from nba_model.features import TARGETS, assemble, feature_columns

VALID_SEASON, TEST_SEASON = 2024, 2025
MIN_HISTORY = 5  # only train/evaluate on games where the player has >=5 prior games
STEP, MAX_ITER = 50, 800


def make_model(target: str, n_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error" if target == "min" else "poisson",
        learning_rate=0.05,
        max_iter=n_iter,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def pick_iterations(target, Xtr, ytr, Xva, yva):
    m = make_model(target, STEP)
    m.set_params(warm_start=True)
    best, best_mae, n = STEP, np.inf, STEP
    while n <= MAX_ITER:
        m.set_params(max_iter=n)
        m.fit(Xtr, ytr)
        mae = np.mean(np.abs(m.predict(Xva) - yva))
        if mae < best_mae - 1e-4:
            best, best_mae = n, mae
        elif n - best >= 3 * STEP:
            break
        n += STEP
    return best


def interval_table(pred, actual, n_bins=10):
    """Empirical 10th/90th percentile of actual outcomes, by predicted-value decile."""
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        r = (actual - pred)[idx == b]
        rows.append((edges[b], float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-data", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress live, even through tee/pipes

    t0 = time.time()
    pg, tg, info = prepare(force=args.rebuild_data)
    X = assemble(pg, tg, info)
    X = X[X.career_games >= MIN_HISTORY].reset_index(drop=True)
    feats = feature_columns(X)
    X[feats] = X[feats].astype("float32")
    print(f"{len(X):,} rows x {len(feats)} features  ({time.time() - t0:.0f}s)")

    tr, va, te = X.season <= VALID_SEASON - 1, X.season == VALID_SEASON, X.season >= TEST_SEASON
    report, models, intervals, n_iters = {}, {}, {}, {}
    for t in TARGETS:
        t1 = time.time()
        n = pick_iterations(t, X.loc[tr, feats], X.loc[tr, t], X.loc[va, feats], X.loc[va, t])
        # honest test: fit on train+valid, score on test
        m = make_model(t, n).fit(X.loc[tr | va, feats], X.loc[tr | va, t])
        pred, y = m.predict(X.loc[te, feats]), X.loc[te, t].to_numpy()
        intervals[t] = interval_table(pred, y)
        lo = np.array([intervals[t][max(0, np.searchsorted([e for e, _, _ in intervals[t]], p, "right") - 1)][1] for p in pred])
        hi = np.array([intervals[t][max(0, np.searchsorted([e for e, _, _ in intervals[t]], p, "right") - 1)][2] for p in pred])
        report[t] = {
            "model_mae": float(np.mean(np.abs(pred - y))),
            "last10_avg_mae": float(np.nanmean(np.abs(X.loc[te, f"{t}_l10"] - y))),
            "season_avg_mae": float(np.nanmean(np.abs(X.loc[te, f"{t}_szn"].fillna(X.loc[te, f"{t}_l10"]) - y))),
            "model_r2": float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
            "interval_80_coverage": float(np.mean((y >= pred + lo) & (y <= pred + hi))),
            "n_iter": n,
        }
        n_iters[t] = n
        print(f"{t:>4}: iters={n:<4} MAE model={report[t]['model_mae']:.3f}  "
              f"last10={report[t]['last10_avg_mae']:.3f}  season={report[t]['season_avg_mae']:.3f}  "
              f"R2={report[t]['model_r2']:.3f}  80%-int cover={report[t]['interval_80_coverage']:.2f}  "
              f"({time.time() - t1:.0f}s)")

    print("\nRefitting on all seasons for deployment...")
    for t in TARGETS:
        models[t] = make_model(t, n_iters[t]).fit(X[feats], X[t])

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"models": models, "features": feats, "intervals": intervals}, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved artifacts/model.joblib and artifacts/metrics.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
