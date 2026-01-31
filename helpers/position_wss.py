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
        self.condition_ids: List[str] = list(condition_ids or [])
        self._pending_condition_ids: List[str] = []
        self.running = False
        self._fill_callbacks: List[Callable[[FillData], None]] = []
        self._order_callbacks: List[Callable[[OrderEventData], None]] = []
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._auth: Optional[_AuthDict] = None

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

    def set_condition_ids(self, condition_ids: List[str]):
        """Update condition IDs to subscribe to (e.g. active markets). Applied on (re)connect."""
        self.condition_ids = list(condition_ids)
        self._pending_condition_ids = list(condition_ids)

    def add_condition_ids(self, condition_ids: List[str]):
        """Add condition IDs; no resubscribe on fly for user channel (reconnect to apply)."""
        for cid in condition_ids:
            if cid not in self.condition_ids:
                self.condition_ids.append(cid)
                self._pending_condition_ids.append(cid)

    def on_fill(self, callback: Callable[[FillData], None]):
        """Register callback for trade fills (event_type trade, status MATCHED)."""
        self._fill_callbacks.append(callback)

    def on_order(self, callback: Callable[[OrderEventData], None]):
        """Register callback for order events (PLACEMENT / UPDATE / CANCELLATION)."""
        self._order_callbacks.append(callback)

    async def stream(self):
        """Connect to User Channel and process messages. Exits when stop() is called."""
        if not self._ensure_auth():
            print("[Position WSS] Missing CLOB_API_KEY / CLOB_SECRET / CLOB_PASS_PHRASE; skipping user channel")
            return

        self.running = True
        while self.running:
            try:
                async with websockets.connect(USER_CHANNEL_WSS) as ws:
                    print("✓ Connected to Polymarket User Channel (position/order stream)")

                    # Subscribe with auth; empty markets = all user activity
                    markets = list(self.condition_ids) if self.condition_ids else []
                    msg = {
                        "markets": markets,
                        "type": "user",
                        "auth": self._auth,
                    }
                    await ws.send(json.dumps(msg))
                    if markets:
                        print(f"  [User] Subscribed to {len(markets)} condition(s)")
                    else:
                        print("  [User] Subscribed to all user activity")

                    self._ws = ws
                    self._pending_condition_ids.clear()

                    while self.running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            # Ping to keep connection alive
                            await ws.ping()
                            continue

                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(data, dict):
                            self._handle_message(data)
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    self._handle_message(item)

                    self._ws = None

            except Exception as e:
                print(f"[Position WSS] Error: {e}, reconnecting...")
                await asyncio.sleep(2)

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
