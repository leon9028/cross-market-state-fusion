#!/usr/bin/env python3
"""
Offline signal check: do entry features (f0..f23, same as MarketState.to_features)
help predict win/loss or PnL on closed trades?

Depends only on NumPy (no scikit-learn). If you use torch/mlx, numpy is usually already installed.

Usage (from repo root):
  python offline_signal_eval.py logs/trades_YYYYMMDD_HHMMSS.csv

Paste the printed block back for analysis.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError:
    print("Missing numpy. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

from strategies.base import STATE_FEATURE_DIM  # noqa: E402


def _load(path: Path):
    rows = []
    skipped_no_features = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty or invalid CSV: {path}")
        if "f0" not in reader.fieldnames:
            raise SystemExit(
                "This CSV has no f0..f23 columns. "
                "Train with the updated logger (entry snapshot) and use a new trades_*.csv."
            )
        for row in reader:
            if not row.get("f0", "").strip():
                skipped_no_features += 1
                continue
            try:
                x = [float(row[f"f{i}"]) for i in range(STATE_FEATURE_DIM)]
                pnl = float(row["pnl"])
            except (KeyError, ValueError):
                skipped_no_features += 1
                continue
            rows.append((row.get("timestamp", ""), x, pnl))
    return rows, skipped_no_features


def _standardize(X_train, X_val, eps: float = 1e-8):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + eps
    return (X_train - mu) / sigma, (X_val - mu) / sigma


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _fit_logistic_balanced(X, y, n_iter: int = 2500, lr: float = 0.15, l2: float = 1e-3):
    """L2 logistic regression with class-balanced sample weights (approx sklearn)."""
    n, d = X.shape
    y = y.astype(np.float64)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(n - y.sum()), 1.0)
    sw = np.where(y == 1.0, n / (2.0 * n_pos), n / (2.0 * n_neg))
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    w = np.zeros(d + 1, dtype=np.float64)
    for _ in range(n_iter):
        p = _sigmoid(Xb @ w)
        err = (p - y) * sw
        grad = (Xb.T @ err) / n
        grad[:d] += l2 * w[:d]
        w -= lr * grad
    return w


def _predict_proba_logistic(X, w):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return _sigmoid(Xb @ w)


def _fit_ridge(X, y, alpha: float = 1.0):
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    reg = np.eye(d + 1, dtype=np.float64) * alpha
    reg[-1, -1] = 0.0
    A = Xb.T @ Xb + reg
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def _predict_ridge(X, w):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ w


def _roc_auc_trapezoid(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_true) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])
    return float(np.trapz(tpr, fpr))


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    mu = np.mean(y_true)
    ss_tot = np.sum((y_true - mu) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate entry features vs trade PnL (offline).")
    ap.add_argument(
        "trades_csv",
        type=Path,
        help="Path to trades_*.csv (with f0..f23 columns from training logger).",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Fraction of rows (time-ordered) for training (default 0.7).",
    )
    args = ap.parse_args()

    path = args.trades_csv
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows, skipped = _load(path)
    n = len(rows)
    if n < 50:
        print(
            f"Too few rows with features: {n} (skipped {skipped} without f0..). "
            "Collect more trades with RL training, then re-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    rows.sort(key=lambda r: r[0])
    split = max(1, int(n * args.train_frac))
    train, val = rows[:split], rows[split:]
    X_train = np.array([r[1] for r in train], dtype=np.float64)
    X_val = np.array([r[1] for r in val], dtype=np.float64)
    y_pnl_train = np.array([r[2] for r in train], dtype=np.float64)
    y_pnl_val = np.array([r[2] for r in val], dtype=np.float64)
    y_cls_train = (y_pnl_train > 0).astype(np.int32)
    y_cls_val = (y_pnl_val > 0).astype(np.int32)

    X_train_s, X_val_s = _standardize(X_train, X_val)

    maj = int(y_cls_train.mean() >= 0.5)
    acc_base = float((y_cls_val == maj).mean())
    pnl_mean_base = float(y_pnl_train.mean())

    w_log = _fit_logistic_balanced(X_train_s, y_cls_train)
    prob_val = _predict_proba_logistic(X_val_s, w_log)
    pred_cls = (prob_val >= 0.5).astype(np.int32)
    acc = float((y_cls_val == pred_cls).mean())
    auc = _roc_auc_trapezoid(y_cls_val, prob_val)

    w_ridge = _fit_ridge(X_train_s, y_pnl_train, alpha=1.0)
    pred_pnl = _predict_ridge(X_val_s, w_ridge)
    r2 = _r2(y_pnl_val, pred_pnl)

    order = np.argsort(prob_val)
    deciles = np.array_split(order, 10)
    dec_means = [float(y_pnl_val[d].mean()) for d in deciles if len(d) > 0]
    top_mean = dec_means[-1] if dec_means else float("nan")
    bot_mean = dec_means[0] if dec_means else float("nan")

    lines = [
        "=" * 60,
        f"file: {path}",
        f"rows_with_features: {n}  skipped_no_f0: {skipped}  train/val: {len(train)}/{len(val)}",
        f"class_balance_val: P(win)={y_cls_val.mean():.3f}",
        "--- classification (win = pnl > 0) ---",
        f"accuracy_majority_baseline: {acc_base:.4f}",
        f"accuracy_logistic:          {acc:.4f}",
        f"roc_auc_logistic:           {auc:.4f}",
        "--- regression (pnl) ---",
        f"mean_pnl_train (constant guess): {pnl_mean_base:+.4f}",
        f"ridge_r2_val:                     {r2:+.4f}",
        "--- decile mean PnL (val, by predicted P(win), low→high) ---",
        "  " + " | ".join(f"{m:+.2f}" for m in dec_means),
        f"  bottom_decile_mean_pnl: {bot_mean:+.4f}  top_decile_mean_pnl: {top_mean:+.4f}",
        "=" * 60,
        "Interpret: AUC ~0.5 and flat deciles => no usable linear signal in these features.",
        "Positive R2 with negative overall PnL can still mean 'explains variance' not edge.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
