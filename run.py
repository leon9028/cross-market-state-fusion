#!/usr/bin/env python3
"""
Unified runner for Polymarket trading strategies.

Usage:
    python run.py                     # List available strategies
    python run.py random              # Run random baseline (paper)
    python run.py mean_revert         # Run mean reversion
    python run.py rl                  # Run RL strategy
    python run.py gating              # Run gating (MoE)
    python run.py rl --train          # Train RL strategy
    python run.py rl --train --dashboard  # Train with web dashboard
    python run.py rl --live       # Run with real orders (requires .env CLOB creds)
"""
import asyncio
import argparse
import copy
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, ".")
from helpers import (
    get_15m_markets,
    BinanceStreamer,
    OrderbookStreamer,
    Market,
    FuturesStreamer,
    get_logger,
    PositionStreamer,
    FillData,
)
from helpers.clob_client import make_client, create_and_submit_order
from py_clob_client.clob_types import OrderType
from strategies import (
    Strategy, MarketState, Action,
    create_strategy, AVAILABLE_STRATEGIES,
    RLStrategy,
)

# Dashboard integration (optional)
try:
    from dashboard_cinematic import update_dashboard_state, update_rl_metrics, emit_rl_buffer, run_dashboard, emit_trade
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    def update_dashboard_state(**kwargs): pass
    def update_rl_metrics(metrics): pass
    def emit_rl_buffer(buffer_size, max_buffer=256, avg_reward=None): pass
    def emit_trade(action, asset, size=0, pnl=None): pass


@dataclass
class Position:
    """Track position for a market."""
    asset: str
    side: Optional[str] = None
    size: float = 0.0  # USD notional (paper) or for display; live close uses shares
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0
    shares: float = 0.0  # Filled shares from User Channel (live); SELL order uses this


class TradingEngine:
    """
    Paper or live trading engine with strategy harness.
    Use --live to place real orders via CLOB; positions then update from User Channel fills.
    """

    def __init__(self, strategy: Strategy, trade_size: float = 10.0, live: bool = False):
        self.strategy = strategy
        self.trade_size = trade_size
        self.live = live
        self.clob_client = None
        if live:
            try:
                self.clob_client = make_client()
                print("  [LIVE] CLOB client ready; orders will be sent to Polymarket.")
            except Exception as e:
                print(f"  [LIVE] CLOB client failed: {e}; falling back to paper trading.")
                self.live = False

        # Streamers
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.position_streamer = PositionStreamer(condition_ids=[])
        self.position_streamer.on_fill(self._on_fill)

        # State
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        self.prev_states: Dict[str, MarketState] = {}  # For RL transitions
        self.open_prices: Dict[str, float] = {}  # Binance price at market open
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # Pending rewards for RL (set on position close)
        self.pending_rewards: Dict[str, float] = {}

        # Logger (for RL training)
        self.logger = get_logger() if isinstance(strategy, RLStrategy) else None

    def refresh_markets(self):
        """Find active 15-min markets."""
        print("\n" + "=" * 60)
        print(f"STRATEGY: {self.strategy.name.upper()}")
        print("=" * 60)

        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)

        # Clear stale data
        self.markets.clear()
        self.states.clear()

        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5:
                continue

            print(f"\n{m.asset} 15m | {mins_left:.1f}m left")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")

            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            # Init state
            self.states[m.condition_id] = MarketState(
                asset=m.asset,
                prob=m.price_up,
                time_remaining=mins_left / 15.0,
            )

            # Init position
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            # Record open price
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

        if not self.markets:
            print("\nNo active markets!")
        else:
            # Clear stale orderbook subscriptions
            active_cids = set(self.markets.keys())
            self.orderbook_streamer.clear_stale(active_cids)
            # User Channel: subscribe to active markets for fill updates
            self.position_streamer.set_condition_ids(list(self.markets.keys()))

    def execute_action(self, cid: str, action: Action, state: MarketState):
        """Execute paper trade or live order (when --live and CLOB client ready)."""
        if action == Action.HOLD:
            return

        pos = self.positions.get(cid)
        if not pos:
            return

        price = state.prob
        trade_amount = self.trade_size * action.size_multiplier
        size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(action.size_multiplier, "")

        # Live: place real order via CLOB; position will update from _on_fill
        if self.live and self.clob_client:
            try:
                self._execute_live_order(cid, pos, action, state, price, trade_amount, size_label)
            except Exception as e:
                print(f"    [LIVE] Order failed: {e}")
            return
        # Paper: update positions in memory
        self._execute_paper_action(cid, pos, action, state, price, trade_amount, size_label)

    def _execute_live_order(
        self, cid: str, pos, action: Action, state: MarketState,
        price: float, trade_amount: float, size_label: str,
    ):
        """Place real order via CLOB. Position updates come from User Channel (_on_fill)."""
        m = self.markets.get(cid)
        if not m:
            return
        ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
        ob_down = self.orderbook_streamer.get_orderbook(cid, "DOWN")

        # Close existing position: SELL uses stored shares (from User Channel matched message)
        if pos.size > 0 or pos.shares > 0:
            # Use pos.shares (filled shares from User Channel); fallback to size/entry_price for paper/migration
            shares_to_sell = pos.shares if pos.shares > 0 else (pos.size / pos.entry_price if pos.entry_price else 0)
            shares_to_sell = float(int(shares_to_sell * 10) / 10)  # round down to 1 decimal (truncate)
            if shares_to_sell <= 0:
                return
            # Strategy says SELL (sell UP) and we hold UP → close UP position by selling token_up
            if action.is_sell and pos.side == "UP":
                token_id = m.token_up
                side = "SELL"
                order_price = (ob_up.best_bid if ob_up and ob_up.best_bid is not None else price)
                print(f"    [LIVE] SUBMIT CLOSE UP {pos.asset} {side} {shares_to_sell:.1f} @ {order_price:.3f}")
                create_and_submit_order(
                    self.clob_client, token_id, side, order_price, shares_to_sell, order_type=OrderType.FAK
                )
                return
            # Strategy says BUY (buy UP) and we hold DOWN → close DOWN position by selling token_down
            elif action.is_buy and pos.side == "DOWN":
                token_id = m.token_down
                side = "SELL"
                down_price = 1 - price
                order_price = (ob_down.best_bid if ob_down and ob_down.best_bid is not None else down_price)
                print(f"    [LIVE] SUBMIT CLOSE DOWN {pos.asset} {side} {shares_to_sell:.1f} @ {order_price:.3f}")
                create_and_submit_order(
                    self.clob_client, token_id, side, order_price, shares_to_sell, order_type=OrderType.FAK
                )
                return

        # Open new position: BUY token_id. API expects size in shares; notional = size * price >= $1.
        if pos.size == 0:
            if action.is_buy:
                token_id = m.token_up
                side = "BUY"
                order_price = (ob_up.best_ask if ob_up and ob_up.best_ask is not None else price)
                if not order_price or order_price <= 0:
                    return
                # Size in shares so notional = trade_amount; min notional $1
                size_shares = trade_amount / order_price
                size_shares = float(int(size_shares * 10) / 10)  # 1 decimal, truncate
                if size_shares < 1.0 / order_price:  # below min $1
                    return
                print(f"    [LIVE] SUBMIT OPEN UP {pos.asset} ({size_label}) {side} {size_shares:.1f} shares @ {order_price:.3f} (${trade_amount:.0f})")
                create_and_submit_order(
                    self.clob_client, token_id, side, order_price, size_shares, order_type=OrderType.FAK
                )
            elif action.is_sell:
                token_id = m.token_down
                side = "BUY"
                down_price = 1 - price
                order_price = (ob_down.best_ask if ob_down and ob_down.best_ask is not None else down_price)
                if not order_price or order_price <= 0:
                    return
                size_shares = trade_amount / order_price
                size_shares = float(int(size_shares * 10) / 10)
                if size_shares < 1.0 / order_price:
                    return
                print(f"    [LIVE] SUBMIT OPEN DOWN {pos.asset} ({size_label}) {side} {size_shares:.1f} shares @ {order_price:.3f} (${trade_amount:.0f})")
                create_and_submit_order(
                    self.clob_client, token_id, side, order_price, size_shares, order_type=OrderType.FAK
                )

    def _execute_paper_action(
        self, cid: str, pos, action: Action, state: MarketState,
        price: float, trade_amount: float, size_label: str,
    ):
        """Execute paper trade (update positions in memory only)."""
        # Close existing position if switching sides
        if pos.size > 0:
            if action.is_sell and pos.side == "UP":
                shares = pos.size / pos.entry_price
                pnl = (price - pos.entry_price) * shares
                self._record_trade(pos, price, pnl, "CLOSE UP", cid=cid)
                self.pending_rewards[cid] = pnl
                pos.size = 0
                pos.side = None
                return
            elif action.is_buy and pos.side == "DOWN":
                exit_down_price = 1 - price
                shares = pos.size / pos.entry_price
                pnl = (exit_down_price - pos.entry_price) * shares
                self._record_trade(pos, price, pnl, "CLOSE DOWN", cid=cid)
                self.pending_rewards[cid] = pnl
                pos.size = 0
                pos.side = None
                return

        # Open new position
        if pos.size == 0:
            if action.is_buy:
                pos.side = "UP"
                pos.size = trade_amount
                pos.entry_price = price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                print(f"    OPEN {pos.asset} UP ({size_label}) ${trade_amount:.0f} @ {price:.3f}")
                emit_trade(f"BUY_{size_label}", pos.asset, pos.size)
            elif action.is_sell:
                pos.side = "DOWN"
                pos.size = trade_amount
                pos.entry_price = 1 - price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                print(f"    OPEN {pos.asset} DOWN ({size_label}) ${trade_amount:.0f} @ {1 - price:.3f}")
                emit_trade(f"SELL_{size_label}", pos.asset, pos.size)

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        """Record completed trade."""
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | PnL: ${pnl:+.2f}")
        # Emit to dashboard
        emit_trade(action, pos.asset, pos.size, pnl)

        # Log to CSV
        if self.logger and pos.entry_time:
            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            binance_change = 0.0
            if cid and cid in self.open_prices:
                current = self.price_streamer.get_price(pos.asset)
                if current > 0 and self.open_prices[cid] > 0:
                    binance_change = (current - self.open_prices[cid]) / self.open_prices[cid]

            self.logger.log_trade(
                asset=pos.asset,
                action="BUY" if "UP" in action else "SELL",
                side=pos.side or "UNKNOWN",
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                duration_sec=duration,
                time_remaining=pos.time_remaining_at_entry,
                prob_at_entry=pos.entry_prob,
                prob_at_exit=price,
                binance_change=binance_change,
                condition_id=cid
            )

    def _compute_step_reward(self, cid: str, state: MarketState, action: Action, pos: Position) -> float:
        """Compute reward signal for RL training - pure realized PnL."""
        # Only reward on position close - cleaner signal
        # Reward is set when trade closes in _execute_trade via self.pending_rewards
        return self.pending_rewards.pop(cid, 0.0)

    def _resolve_condition_id(self, raw_cid: str) -> Optional[str]:
        """Resolve fill condition_id to engine's position key (0x or no-0x)."""
        if not raw_cid:
            return None
        if raw_cid in self.positions:
            return raw_cid
        if raw_cid.startswith("0x") and raw_cid[2:] in self.positions:
            return raw_cid[2:]
        if not raw_cid.startswith("0x") and f"0x{raw_cid}" in self.positions:
            return f"0x{raw_cid}"
        return None

    def _on_fill(self, fill: FillData):
        """Update positions from User Channel fill (status MATCHED).
        BUY order uses USD; User Channel matched message gives filled shares.
        We store pos.shares = filled shares; SELL (close) uses pos.shares for order size.
        """
        cid = self._resolve_condition_id(fill.condition_id)
        if not cid:
            return
        pos = self.positions.get(cid)
        if not pos:
            return
        state = self.states.get(cid)
        time_remaining = state.time_remaining if state else 0.0

        if fill.side == "BUY":
            # fill.size = filled shares (tokens) from User Channel
            filled_shares = fill.size
            filled_usd = filled_shares * fill.price if fill.price and fill.price > 0 else 0.0
            if pos.size <= 0 and pos.shares <= 0:
                pos.side = fill.outcome
                pos.entry_price = fill.price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = fill.price if fill.outcome == "UP" else (1.0 - fill.price)
                pos.time_remaining_at_entry = time_remaining
                pos.shares = filled_shares
                pos.size = filled_usd
                print(f"    [FILL] OPEN {pos.asset} {fill.outcome} {filled_shares:.4f} shares @ {fill.price:.3f} (${filled_usd:.0f})")
            else:
                # Add to position (same side): VWAP
                old_shares = pos.shares
                old_usd = pos.size
                pos.shares += filled_shares
                pos.size += filled_usd
                pos.entry_price = pos.size / pos.shares if pos.shares > 0 else fill.price
                print(f"    [FILL] ADD {pos.asset} {fill.outcome} +{filled_shares:.4f} shares @ {fill.price:.3f} -> {pos.shares:.4f} shares (${pos.size:.0f})")
        else:
            # SELL: close or reduce (fill.size = filled shares)
            if pos.shares <= 0 and pos.size <= 0:
                return
            filled_shares = fill.size
            pos.shares -= filled_shares
            if pos.shares <= 0:
                pos.shares = 0.0
                closed_side = pos.side
                if closed_side == "UP":
                    pnl = (fill.price - pos.entry_price) * filled_shares
                else:
                    pnl = ((1.0 - fill.price) - pos.entry_price) * filled_shares
                pos.size = 0
                pos.side = None
                pos.size = filled_shares * fill.price  # for _record_trade emit
                self._record_trade(pos, fill.price, pnl, f"CLOSE {closed_side}", cid=cid)
                pos.size = 0
                self.pending_rewards[cid] = pnl
                print(f"    [FILL] CLOSE {pos.asset} {filled_shares:.4f} shares @ {fill.price:.3f} | PnL: ${pnl:+.2f}")
            else:
                pos.size = pos.shares * pos.entry_price  # keep USD in sync
                print(f"    [FILL] REDUCE {pos.asset} {pos.side} -{filled_shares:.4f} shares -> {pos.shares:.4f} shares (${pos.size:.0f})")

    def close_all_positions(self):
        """Close all positions at current prices (in-memory only; live positions need SELL order)."""
        for cid, pos in self.positions.items():
            if pos.size > 0 or pos.shares > 0:
                state = self.states.get(cid)
                if state:
                    price = state.prob
                    shares = pos.shares if pos.shares > 0 else (pos.size / pos.entry_price if pos.entry_price else 0)
                    if pos.side == "UP":
                        pnl = (price - pos.entry_price) * shares
                    else:
                        exit_down_price = 1 - price
                        pnl = (exit_down_price - pos.entry_price) * shares

                    self._record_trade(pos, price, pnl, f"FORCE CLOSE {pos.side}", cid=cid)
                    self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                    pos.size = 0
                    pos.shares = 0.0
                    pos.side = None

    async def decision_loop(self):
        """Main trading loop."""
        tick = 0
        tick_interval = 0.5  # 500ms ticks for faster decisions
        while self.running:
            await asyncio.sleep(tick_interval)
            tick += 1
            now = datetime.now(timezone.utc)

            # Check expired markets
            expired = [cid for cid, m in self.markets.items() if m.end_time <= now]
            for cid in expired:
                print(f"\n  EXPIRED: {self.markets[cid].asset}")

                # RL: Store terminal experience with final PnL
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    state = self.states.get(cid)
                    prev_state = self.prev_states.get(cid)
                    pos = self.positions.get(cid)
                    if state and prev_state:
                        # Terminal reward is the realized PnL
                        terminal_reward = state.position_pnl if pos and (pos.size > 0 or pos.shares > 0) else 0.0
                        self.strategy.store(prev_state, Action.HOLD, terminal_reward, state, done=True)

                    # Clean up prev_state
                    if cid in self.prev_states:
                        del self.prev_states[cid]

                del self.markets[cid]

            if not self.markets:
                print("\nAll markets expired. Refreshing...")
                self.close_all_positions()
                self.refresh_markets()
                if not self.markets:
                    print("No new markets. Waiting...")
                    await asyncio.sleep(30)
                continue

            # Update states and make decisions
            for cid, m in self.markets.items():
                state = self.states.get(cid)
                if not state:
                    continue

                # Update state from orderbook - CRITICAL for 15-min
                ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                if ob and ob.mid_price:
                    state.prob = ob.mid_price
                    state.prob_history.append(ob.mid_price)
                    if len(state.prob_history) > 100:
                        state.prob_history = state.prob_history[-100:]
                    state.best_bid = ob.best_bid or 0.0
                    state.best_ask = ob.best_ask or 0.0
                    state.spread = ob.spread or 0.0

                    # Orderbook imbalance - L1 (top of book)
                    if ob.bids and ob.asks:
                        bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                        ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                        total_l1 = bid_vol_l1 + ask_vol_l1
                        state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

                        # Orderbook imbalance - L5 (depth)
                        bid_vol_l5 = sum(size for _, size in ob.bids[:5])
                        ask_vol_l5 = sum(size for _, size in ob.asks[:5])
                        total_l5 = bid_vol_l5 + ask_vol_l5
                        state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0

                # Update binance price
                binance_price = self.price_streamer.get_price(m.asset)
                state.binance_price = binance_price
                open_price = self.open_prices.get(cid, binance_price)
                if open_price > 0:
                    state.binance_change = (binance_price - open_price) / open_price

                # Update futures data (focused on fast-updating features)
                futures = self.futures_streamer.get_state(m.asset)
                if futures:
                    # Order flow - THE EDGE
                    old_cvd = state.cvd
                    state.cvd = futures.cvd
                    state.cvd_acceleration = (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
                    state.trade_flow_imbalance = futures.trade_flow_imbalance

                    # Ultra-short momentum
                    state.returns_1m = futures.returns_1m
                    state.returns_5m = futures.returns_5m
                    state.returns_10m = futures.returns_10m  # Properly computed from klines

                    # Microstructure - CRITICAL for 15-min
                    state.trade_intensity = futures.trade_intensity
                    state.large_trade_flag = futures.large_trade_flag

                    # Volatility
                    state.realized_vol_5m = futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
                    state.vol_expansion = futures.vol_ratio - 1.0

                    # Regime context (slow but useful for context)
                    state.vol_regime = 1.0 if futures.realized_vol_1h > 0.01 else 0.0
                    state.trend_regime = 1.0 if abs(futures.returns_1h) > 0.005 else 0.0

                # Time remaining - CRITICAL
                state.time_remaining = (m.end_time - now).total_seconds() / 900

                # Update position info in state
                pos = self.positions.get(cid)
                if pos and (pos.size > 0 or pos.shares > 0):
                    state.has_position = True
                    state.position_side = pos.side
                    shares = pos.shares if pos.shares > 0 else (pos.size / pos.entry_price if pos.entry_price else 0)
                    if pos.side == "UP":
                        state.position_pnl = (state.prob - pos.entry_price) * shares
                    else:
                        current_down_price = 1 - state.prob
                        state.position_pnl = (current_down_price - pos.entry_price) * shares
                else:
                    state.has_position = False
                    state.position_side = None
                    state.position_pnl = 0.0

                # For non-RL strategies, force close near expiry as safety
                # For RL, let it learn to close on its own (gets penalty at expiry)
                if pos and (pos.size > 0 or pos.shares > 0) and state.very_near_expiry:
                    if not isinstance(self.strategy, RLStrategy):
                        print(f"    ⏰ EARLY CLOSE: {pos.asset}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue

                # Get action from strategy
                action = self.strategy.act(state)

                # RL: Store experience EVERY tick (dense learning signal)
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    prev_state = self.prev_states.get(cid)
                    if prev_state:
                        step_reward = self._compute_step_reward(cid, state, action, pos)
                        # Episode not done unless market expired
                        self.strategy.store(prev_state, action, step_reward, state, done=False)

                    # Deep copy state for next iteration
                    self.prev_states[cid] = copy.deepcopy(state)

                # Execute
                if action != Action.HOLD:
                    self.execute_action(cid, action, state)

            # Status update every 10 ticks (console), but dashboard every tick
            if tick % 10 == 0:
                self.print_status()
            else:
                # Update dashboard state every tick for responsiveness
                self._update_dashboard_only()

            # RL training: emit buffer progress every tick
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                buffer_size = len(self.strategy.experiences)
                # Compute average reward from recent experiences
                avg_reward = None
                if buffer_size > 0:
                    recent_rewards = [exp.reward for exp in self.strategy.experiences[-50:]]  # Last 50
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                emit_rl_buffer(buffer_size, self.strategy.buffer_size, avg_reward)

                # PPO update when buffer is full
                if buffer_size >= self.strategy.buffer_size:
                    # Get buffer rewards before update clears them
                    buffer_rewards = [exp.reward for exp in self.strategy.experiences]
                    metrics = self.strategy.update()
                    if metrics:
                        print(f"  [RL] loss={metrics['policy_loss']:.4f} "
                              f"v_loss={metrics['value_loss']:.4f} "
                              f"ent={metrics['entropy']:.3f} "
                              f"kl={metrics['approx_kl']:.4f} "
                              f"ev={metrics['explained_variance']:.2f}")
                        # Send to dashboard
                        metrics['buffer_size'] = len(self.strategy.experiences)
                        update_rl_metrics(metrics)
                        # Log to CSV
                        if self.logger:
                            self.logger.log_update(
                                metrics=metrics,
                                buffer_rewards=buffer_rewards,
                                cumulative_pnl=self.total_pnl,
                                cumulative_trades=self.trade_count,
                                cumulative_wins=self.win_count
                            )

    def _update_dashboard_only(self):
        """Update dashboard state without printing to console."""
        now = datetime.now(timezone.utc)
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                vel = state._velocity()
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        win_rate = self.win_count / max(1, self.trade_count) * 100

        print(f"\n[{now.strftime('%H:%M:%S')}] {self.strategy.name.upper()}")
        print(f"  PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%")

        # Prepare dashboard data
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                pos_str = f"{pos.side} ${pos.size:.0f}" if pos and (pos.size > 0 or pos.shares > 0) else "FLAT"
                vel = state._velocity()
                print(f"  {m.asset}: prob={state.prob:.3f} vel={vel:+.3f} | {pos_str} | {mins_left:.1f}m")

                # Dashboard data
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        # Update dashboard
        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_final_stats(self):
        """Print final results."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trade_count}")
        print(f"Win Rate: {self.win_count / max(1, self.trade_count) * 100:.1f}%")

    async def run(self):
        """Run the trading engine."""
        self.running = True
        self.refresh_markets()

        if not self.markets:
            print("No markets to trade!")
            return

        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.position_streamer.stream(),
            self.decision_loop(),
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass  # Handle in finally
        finally:
            print("\n\nShutting down...")
            self.running = False
            self.price_streamer.stop()
            self.orderbook_streamer.stop()
            self.futures_streamer.stop()
            self.position_streamer.stop()
            self.close_all_positions()
            self.print_final_stats()

            # Save RL model if training
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                self.strategy.save("rl_model")
                print("  [RL] Model saved to rl_model.pt / rl_model.safetensors (+ _stats.npz)")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument(
        "strategy",
        nargs="?",
        choices=AVAILABLE_STRATEGIES,
        help="Strategy to run"
    )
    parser.add_argument("--train", action="store_true", help="Enable training mode for RL")
    parser.add_argument("--size", type=float, default=10.0, help="Trade size in $")
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--live", action="store_true", help="Place real orders via CLOB (requires .env: PK, FUNDER, CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE)")

    args = parser.parse_args()

    if not args.strategy:
        print("Available strategies:")
        for name in AVAILABLE_STRATEGIES:
            print(f"  - {name}")
        print("\nUsage: python run.py <strategy>")
        print("       python run.py rl --train")
        print("       python run.py rl --train --dashboard")
        print("       python run.py random --live   # real orders (requires .env CLOB creds)")
        return

    # Start dashboard in background if requested
    if args.dashboard:
        if DASHBOARD_AVAILABLE:
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                kwargs={'port': args.port},
                daemon=True
            )
            dashboard_thread.start()
            import time
            time.sleep(1)  # Give dashboard time to start
        else:
            print("Warning: Dashboard not available. Install flask-socketio.")

    # Create strategy
    strategy = create_strategy(args.strategy)

    # RL-specific setup
    if isinstance(strategy, RLStrategy):
        if args.load:
            strategy.load(args.load)
            print(f"Loaded RL model from {args.load}")
        if args.train:
            strategy.train()
            print("RL training mode enabled")
        else:
            strategy.eval()

    # Run
    engine = TradingEngine(strategy, trade_size=args.size, live=args.live)
    if args.live:
        print("\n*** LIVE MODE: real orders will be sent to Polymarket ***\n")
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
