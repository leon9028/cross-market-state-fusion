# Cross-Market State Fusion

RL agents that exploit information lag between fast markets (Binance futures) and slow markets (Polymarket prediction markets) through real-time multi-source state fusion.

**[View the presentation (PDF)](cross-market-state-fusion.pdf)** | **[LACUNA visual writeup](https://humanplane.com/lacuna)**

## What This Is

A PPO (Proximal Policy Optimization) agent that paper trades Polymarket's 15-minute binary crypto markets. The agent observes live data from Binance futures and Polymarket's orderbook, then learns to predict short-term price direction.

**Current status**: Paper trading only. The agent trains and makes decisions on live market data, but doesn't execute real orders.

### Setup
- **Markets**: 4 concurrent 15-min binary crypto markets (BTC, ETH, SOL, XRP) on Polymarket
- **Position size**: $5–$500 per trade (configurable via `--size`)
- **Max exposure**: Position size × 4 markets
- **Data sources**: Binance futures (order flow, returns) + Polymarket CLOB (orderbook)
- **Training**: Online PPO with MLX on Apple Silicon, learns from live market data
- **Reward**: Share-based PnL on position close (sparse signal)

## What This Proves

1. **RL can learn from sparse PnL signals** - The agent only gets reward when positions close. No intermediate feedback during the 15-minute window. Despite this sparsity, it learns profitable patterns (~$50K PnL, 2,500% ROI in Phase 5 with temporal architecture).

2. **Multi-source data fusion works** - Combining Binance futures order flow and Polymarket orderbook state into a single 18-dim observation gives the agent useful signal.

3. **Low win rate can be profitable** - The agent wins only 23% of trades but profits because binary markets have asymmetric payoffs. Buy at 0.40, win pays 0.60; lose costs 0.40.

4. **On-device training is viable** - MLX on Apple Silicon handles real-time PPO updates during live market hours without cloud GPU costs.

5. **Temporal context helps** - Processing the last 5 market states through a TemporalEncoder improves decision quality by capturing momentum and trend patterns.

**Important caveat**: Training uses share-based PnL, not actual binary outcomes. See Phase 4 below for why this matters.

## What This Doesn't Prove

1. **Live profitability** - Paper trading assumes instant fills at mid-price. Real trading faces latency, slippage, and market impact. Expect 20-50% performance degradation.

2. **Statistical significance** - A single session isn't enough to confirm edge. Could be variance. Needs weeks of out-of-sample testing.

3. **Scalability** - $500 positions already show some market impact. At larger sizes the agent's orders would move prices and consume liquidity faster than it can trade.

4. **Persistence of edge** - Markets adapt. If this strategy worked, others would copy it and arbitrage it away.

## Path to Live Trading

To move from paper to real:

1. **Execution layer** - Integrate Polymarket CLOB API for order placement
2. **Slippage modeling** - Simulate walking the book at realistic sizes
3. **Latency compensation** - Account for 50-200ms round-trip to Polymarket
4. **Risk management** - Position limits, drawdown stops, exposure caps
5. **Extended validation** - Weeks of paper trading across market regimes

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for detailed training analysis.

---

## Training Evolution

The agent evolved through 5 phases, each fixing problems discovered in the previous:

| Phase | What Changed | Size | PnL | ROI |
|-------|--------------|------|-----|-----|
| 1 | Shaped rewards (failed - entropy collapsed) | $5 | $3.90 | - |
| 2 | Sparse PnL only, simplified actions (7→3) | $5 | $10.93 | 55% |
| 3 | Scaled up 10x | $50 | $23.10 | 12% |
| 4 | Share-based PnL (matches actual market economics) | $500 | $3,392 | 170% |
| 5 | Temporal architecture + feature normalization | $500 | ~$50K | 2,500% |

**Key insight**: Phase 4's switch from `pnl = (exit - entry) × dollars` to `pnl = (exit - entry) × shares` was the breakthrough. At entry price 0.30, you get 3.33 shares per dollar vs 1.43 at 0.70 - same price move, larger return.

**Phase 5 (LACUNA)**: 10+ hour paper trading session. Added TemporalEncoder to capture momentum from last 5 states. BTC carried with +$40K of the $50K total.

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for detailed analysis with charts and code.

---

## Architecture

```
├── run.py                    # Main trading engine
├── dashboard.py              # Real-time web dashboard
├── strategies/
│   ├── base.py               # Action, MarketState, Strategy base classes
│   ├── rl_mlx.py             # PPO implementation (MLX)
│   ├── momentum.py           # Momentum baseline
│   ├── mean_revert.py        # Mean reversion baseline
│   └── fade_spike.py         # Spike fading baseline
└── helpers/
    ├── polymarket_api.py     # Polymarket REST API
    ├── binance_wss.py        # Binance WebSocket (spot price reference)
    ├── binance_futures.py    # Futures data (returns, order flow, CVD)
    └── orderbook_wss.py      # Polymarket CLOB WebSocket
```

## State Space (18 dimensions)

| Category | Features | Source |
|----------|----------|--------|
| Momentum | `returns_1m`, `returns_5m`, `returns_10m` | Binance futures |
| Order Flow | `ob_imbalance_l1`, `ob_imbalance_l5`, `trade_flow`, `cvd_accel` | Binance futures |
| Microstructure | `spread_pct`, `trade_intensity`, `large_trade_flag` | Polymarket CLOB |
| Volatility | `vol_5m`, `vol_expansion` | Polymarket (local), Binance futures |
| Position | `has_position`, `position_side`, `position_pnl`, `time_remaining` | Internal |
| Regime | `vol_regime`, `trend_regime` | Derived |

**Note**: Returns and spread are scaled by 100x. CVD acceleration is divided by 1e6. `vol_5m` is computed locally from Polymarket prob history; `vol_expansion` comes from Binance futures.

## Action Space

| Action | Description |
|--------|-------------|
| HOLD (0) | No action |
| BUY (1) | Long UP token (bet price goes up) |
| SELL (2) | Long DOWN token (bet price goes down) |

Fixed 50% position sizing. Originally had 7 actions with variable sizing (25/50/100%), simplified to reduce complexity.

## Network

```
TemporalEncoder: (5 states × 18 features) → 64 → LayerNorm → tanh → 32

Actor:  [current(18) + temporal(32)] → 64 → LN → tanh → 64 → LN → tanh → 3 (softmax)
Critic: [current(18) + temporal(32)] → 96 → LN → tanh → 96 → LN → tanh → 1
```

- **Temporal processing**: Last 5 states compressed into 32-dim momentum/trend features
- **Asymmetric**: Larger critic (96) vs actor (64) for better value estimation
- **Normalization**: All 18 features clamped to [-1, 1]

## PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr_actor` | 1e-4 | |
| `lr_critic` | 3e-4 | Higher for faster value learning |
| `gamma` | 0.95 | Short horizon (15-min markets) |
| `gae_lambda` | 0.95 | |
| `clip_epsilon` | 0.2 | |
| `entropy_coef` | 0.03 | Low - allow sparse policy (mostly HOLD) |
| `buffer_size` | 256 | Small for faster adaptation |
| `batch_size` | 64 | |
| `n_epochs` | 10 | |
| `history_len` | 5 | Temporal context window |
| `temporal_dim` | 32 | Compressed momentum features |

---

## Usage

```bash
# Training
python run.py --strategy rl --train --size 50

# Inference (load trained model)
python run.py --strategy rl --load rl_model --size 100

# Dashboard (separate terminal)
python dashboard.py --port 5001

# Baselines
python run.py --strategy momentum
python run.py --strategy mean_revert
python run.py --strategy random
```

## Requirements

See `requirements.txt`. On **Apple Silicon (Mac)** the RL strategy uses MLX; on **Linux/Ubuntu/EC2** it uses PyTorch (no MLX). Install with:

```bash
pip install -r requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Running on Ubuntu/EC2:** The RL strategy automatically uses PyTorch when MLX is not available. Train/save on Ubuntu produces `rl_model.pt` and `rl_model_stats.npz`; use `--load rl_model` to load. (Models saved on Mac with MLX use `.safetensors` and are not loadable on Linux without conversion.)

**Low resource (e.g. <2GB RAM / small disk):**
1. Install CPU-only PyTorch first (smaller download and disk use):  
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`  
   Then run `pip install -r requirements.txt` (torch will already be satisfied).
2. When running RL, reduce memory by setting env vars before the command:  
   `export RL_BUFFER_SIZE=128 RL_BATCH_SIZE=32`  
   then e.g. `python run.py rl --train --size 8`. Omit `--dashboard` if RAM is tight.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{cross_market_state_fusion,
  author = {Saravanan, Nikshep},
  organization = {HumanPlane},
  title = {Cross-Market State Fusion: Exploiting Information Lag with Multi-Source RL},
  year = {2025},
  url = {https://github.com/humanplane/cross-market-state-fusion}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
