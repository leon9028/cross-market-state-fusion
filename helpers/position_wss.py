"""
Polymarket CLOB WebSocket User Channel for position/order/trade updates.

Connects to the authenticated User Channel to receive real-time trade and order
events. Use on_fill for confirmed fills (status MATCHED) to sync positions.
"""
import asyncio
import json
import os
import websockets
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Load .env from project root (so it works no matter where you run the script from)
try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

# Type alias to avoid nested-bracket parsing issues in strict checkers
_AuthDict = Dict[str, str]

USER_CHANNEL_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


@dataclass
class FillData:
    """Parsed fill from User Channel trade message (status MATCHED)."""
    condition_id: str
    asset_id: str
    outcome: str   # "UP" or "DOWN" (mapped from YES/NO)
    side: str     # "BUY" or "SELL"
    price: float
    size: float
    trade_id: str
    status: str
    timestamp: Optional[str] = None


@dataclass
class OrderEventData:
    """Parsed order event (PLACEMENT / UPDATE / CANCELLATION)."""
    condition_id: str
    asset_id: str
    outcome: str
    side: str
    price: str
    order_id: str
    event_type: str   # "order"
    type: str         # "PLACEMENT" | "UPDATE" | "CANCELLATION"
    original_size: str = ""
    size_matched: str = "0"
    timestamp: Optional[str] = None


def _outcome_to_side(outcome: str) -> str:
    """Map API outcome YES/NO to internal side UP/DOWN."""
    if (outcome or "").upper() == "YES":
        return "UP"
    if (outcome or "").upper() == "NO":
        return "DOWN"
    return outcome or ""


class PositionStreamer:
    """
    Stream user order and trade events from Polymarket CLOB User Channel.

    Requires env: CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE.
    Optional: pass condition_ids to filter; empty list subscribes to all user activity.
    """

    def __init__(self, condition_ids: Optional[List[str]] = None):
        self.cids: List[str] = list(condition_ids or []) # List of condition IDs to subscribe to
        self._pending_cids: List[str] = [] # New condition IDs to subscribe to (queued)
        self.running = False # Flag to indicate if the streamer is running
        self._fill_callbacks: List[Callable[[FillData], None]] = []
        self._order_callbacks: List[Callable[[OrderEventData], None]] = [] # List of callbacks for order events
        self._ws: Optional[websockets.WebSocketClientProtocol] = None # WebSocket client
        self._auth: Optional[_AuthDict] = None # Authentication dictionary
        self._force_reconnect = False # Flag to trigger reconnection
        self._connected_event: Optional[asyncio.Event] = None # Event to signal when the streamer is connected

    def _ensure_auth(self) -> bool:
        """Build auth dict from env; return True if credentials present."""
        if self._auth is not None:
            return True
        api_key = os.environ.get("CLOB_API_KEY", "").strip()
        secret = os.environ.get("CLOB_SECRET", "").strip()
        passphrase = os.environ.get("CLOB_PASS_PHRASE", "").strip()
        if not (api_key and secret and passphrase):
            return False
        self._auth = {
            "apiKey": api_key,
            "secret": secret,
            "passphrase": passphrase,
        }
        return True

    def clear_stale(self, active_condition_ids: set):
        """Remove condition IDs for expired markets and trigger reconnection."""
        stale_keys = [k for k in self.cids
                    if k not in active_condition_ids]

        had_stale = len(stale_keys) > 0
        for cid in stale_keys:
            self.cids.remove(cid)

        # If we removed subscriptions, force reconnect to cleanly re-subscribe
        # This fixes the issue where User Channel gets stuck on stale condition IDs
        if had_stale:
            print(f"  [User] Cleared {len(stale_keys)} stale condition IDs, triggering reconnect")
            self._force_reconnect = True

    def subscribe(self, condition_id: str):
        """Subscribe to a condition ID."""
        if condition_id not in self.cids:
            self.cids.append(condition_id)
            self._pending_cids.append(condition_id)
            print(f"  [User] Queued {condition_id[:8]}... (pending: {len(self._pending_cids)})")

    def on_fill(self, callback: Callable[[FillData], None]):
        """Register callback for trade fills (event_type trade, status MATCHED)."""
        self._fill_callbacks.append(callback)

    def on_order(self, callback: Callable[[OrderEventData], None]):
        """Register callback for order events (PLACEMENT / UPDATE / CANCELLATION)."""
        self._order_callbacks.append(callback)

    async def wait_connected(self, timeout: Optional[float] = 15.0):
        """Wait until User Channel is connected and subscribed. Used at startup and after rollover before placing orders."""
        if self._connected_event is None:
            return
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print("[Position WSS] wait_connected timed out")

    async def stream(self):
        """Connect to User Channel and process messages. Exits when stop() is called."""
        if not self._ensure_auth():
            print("[Position WSS] Missing CLOB_API_KEY / CLOB_SECRET / CLOB_PASS_PHRASE; skipping user channel")
            return

        self._connected_event = asyncio.Event()
        self.running = True
        while self.running:
            # Wait for subscriptions if none exist yet
            if not self.cids and not self._pending_cids:
                await asyncio.sleep(0.5)
                continue

            try:
                async with websockets.connect(USER_CHANNEL_WSS) as ws:
                    print("✓ Connected to Polymarket User Channel (position/order stream)")

                    # Collect all condition IDs for initial subscription
                    cids = self.cids.copy()

                    # Also include any pending cids
                    if self._pending_cids:
                        # cids.extend(self._pending_cids)
                        self._pending_cids.clear()

                    if cids:
                        # Send single subscription with all condition IDs
                        cids_msg = {
                            "markets": cids,
                            "type": "user",
                            "auth": self._auth,
                        }
                        await ws.send(json.dumps(cids_msg))
                        print(f"  [User] Subscribed to {len(cids)} markets")

                    self._ws = ws
                    self._connected_event.set()

                    # Listen for updates
                    while self.running:
                        try:
                            if self._force_reconnect:
                                print("  [User] Force reconnect triggered, closing connection...")
                                self._force_reconnect = False
                                self._connected_event.clear()
                                break

                            if self._pending_cids:
                                new_cids = self._pending_cids.copy()
                                self._pending_cids.clear()
                                cids_msg = {
                                    "markets": new_cids,
                                    "type": "user",
                                    "auth": self._auth,
                                }
                                await ws.send(json.dumps(cids_msg))
                                print(f"  [User] Sent subscription for {len(new_cids)} new markets")

                            # Short timeout to check pending cids frequently
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                            data = json.loads(msg)

                            # Handle different message types
                            if isinstance(data, dict):
                                self._handle_message(data)
                        except asyncio.TimeoutError:
                            pass
                        except json.JSONDecodeError:
                            pass

                    self._ws = None
                    self._connected_event.clear()

            except Exception as e:
                print(f"CLOB WSS error: {e}, reconnecting...")
                await asyncio.sleep(1)

    def _handle_message(self, data: dict):
        """Dispatch trade vs order messages to callbacks."""
        event_type = (data.get("event_type") or data.get("type") or "").strip().lower()

        if event_type == "trade":
            status = (data.get("status") or "").strip().upper()
            if status == "MATCHED":
                fill = self._parse_fill(data)
                if fill:
                    for cb in self._fill_callbacks:
                        try:
                            cb(fill)
                        except Exception:
                            pass
            return

        if event_type == "order":
            order_evt = self._parse_order(data)
            if order_evt:
                for cb in self._order_callbacks:
                    try:
                        cb(order_evt)
                    except Exception:
                        pass

    def _parse_fill(self, data: dict) -> Optional[FillData]:
        """Build FillData from trade message. Normalize market to condition_id format."""
        market = data.get("market") or data.get("condition_id") or ""
        if not market:
            return None
        # API may return condition_id with or without 0x
        condition_id = market if market.startswith("0x") else market
        try:
            price = float(data.get("price") or 0)
            size = float(data.get("size") or 0)
        except (TypeError, ValueError):
            return None
        outcome_raw = (data.get("outcome") or "").strip().upper()
        side_internal = _outcome_to_side(outcome_raw) if outcome_raw in ("YES", "NO") else outcome_raw
        return FillData(
            condition_id=condition_id,
            asset_id=str(data.get("asset_id") or ""),
            outcome=side_internal,
            side=str(data.get("side") or "BUY").strip().upper(),
            price=price,
            size=size,
            trade_id=str(data.get("id") or data.get("trade_id") or ""),
            status=str(data.get("status") or "MATCHED").strip().upper(),
            timestamp=data.get("timestamp") or data.get("matchtime"),
        )

    def _parse_order(self, data: dict) -> Optional[OrderEventData]:
        """Build OrderEventData from order message."""
        market = data.get("market") or data.get("condition_id") or ""
        if not market:
            return None
        condition_id = market if market.startswith("0x") else market
        evt_type = (data.get("type") or "PLACEMENT").strip().upper()
        return OrderEventData(
            condition_id=condition_id,
            asset_id=str(data.get("asset_id") or ""),
            outcome=_outcome_to_side((data.get("outcome") or "").strip().upper()) or (data.get("outcome") or ""),
            side=str(data.get("side") or "BUY").strip().upper(),
            price=str(data.get("price") or "0"),
            order_id=str(data.get("id") or ""),
            event_type="order",
            type=evt_type,
            original_size=str(data.get("original_size") or "0"),
            size_matched=str(data.get("size_matched") or "0"),
            timestamp=data.get("timestamp"),
        )

    def stop(self):
        """Stop the stream loop."""
        self.running = False


if __name__ == "__main__":
    # Test: connect to User Channel and print any trade/order (requires env credentials)
    async def main():
        api_key = os.environ.get("CLOB_API_KEY", "").strip()
        secret = os.environ.get("CLOB_SECRET", "").strip()
        passphrase = os.environ.get("CLOB_PASS_PHRASE", "").strip()
        if not api_key or not secret or not passphrase:
            print("Set CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE in .env (or export them).")
            print("Example: CLOB_API_KEY=xxx CLOB_SECRET=yyy CLOB_PASS_PHRASE=zzz python helpers/position_wss.py")
            return

        streamer = PositionStreamer(condition_ids=[])

        def on_fill(f: FillData):
            print(f"  FILL: {f.condition_id[:16]}... {f.side} {f.outcome} size={f.size} @ {f.price}")

        def on_order(o: OrderEventData):
            print(f"  ORDER: {o.type} {o.condition_id[:16]}... {o.side} {o.outcome}")

        streamer.on_fill(on_fill)
        streamer.on_order(on_order)

        print("Running User Channel test. Place/cancel an order on Polymarket to see events.")
        print("Press Ctrl+C to stop.\n")
        task = asyncio.create_task(streamer.stream())
        try:
            await task
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            streamer.stop()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            print("\nStopped.")

    asyncio.run(main())
