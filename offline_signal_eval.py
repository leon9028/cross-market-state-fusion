#!/usr/bin/env python3
"""
Offline signal check: do entry features (f0..f23, same as MarketState.to_features)
help predict win/loss or PnL on closed trades?

Usage (from repo root):
  pip install scikit-learn numpy
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

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Missing dependency. Run: pip install scikit-learn numpy", file=sys.stderr)
        sys.exit(1)

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

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Baselines
    maj = int(y_cls_train.mean() >= 0.5)
    acc_base = float((y_cls_val == maj).mean())
    pnl_mean_base = float(y_pnl_train.mean())

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X_train_s, y_cls_train)
    prob_val = clf.predict_proba(X_val_s)[:, 1]
    pred_cls = (prob_val >= 0.5).astype(np.int32)
    acc = accuracy_score(y_cls_val, pred_cls)

    try:
        auc = roc_auc_score(y_cls_val, prob_val)
    except ValueError:
        auc = float("nan")

    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(X_train_s, y_pnl_train)
    pred_pnl = reg.predict(X_val_s)
    r2 = r2_score(y_pnl_val, pred_pnl)

    # Decile lift by predicted win probability (validation)
    order = np.argsort(prob_val)
    deciles = np.array_split(order, 10)
    dec_means = []
    for d in deciles:
        if len(d) == 0:
            continue
        dec_means.append(float(y_pnl_val[d].mean()))
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
    report = "\n".join(lines)
    print(report)


if __name__ == "__main__":
    main()
