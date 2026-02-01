from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.constants import POLYGON
from py_clob_client.order_builder.constants import BUY, SELL
import os
from typing import Literal

try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

CLOB_HOST_DEFAULT = "https://clob.polymarket.com"


def make_client() -> ClobClient:
    host = (os.getenv("CLOB_API_URL") or "").strip() or CLOB_HOST_DEFAULT
    key = (os.getenv("PK") or "").strip()
    funder = (os.getenv("FUNDER") or "").strip()
    api_key = (os.getenv("CLOB_API_KEY") or "").strip()
    api_secret = (os.getenv("CLOB_SECRET") or "").strip()
    api_passphrase = (os.getenv("CLOB_PASS_PHRASE") or "").strip()

    if not key:
        raise ValueError("PK (private key) is required for ClobClient")
    if not api_key or not api_secret or not api_passphrase:
        raise ValueError("CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE are required for L2 (post order)")
    if not funder:
        raise ValueError("FUNDER (proxy/funder address) is required for signature_type=2")

    client = ClobClient(
        host,
        key=key,
        chain_id=POLYGON,
        signature_type=2,
        funder=funder,
        creds=ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        ),
    )
    return client


def _side_to_constant(side: str):
    if (side or "").strip().upper() == "BUY":
        return BUY
    if (side or "").strip().upper() == "SELL":
        return SELL
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def create_and_submit_order(
    client: ClobClient,
    token_id: str,
    side: Literal["BUY", "SELL"],
    price: float,
    size: float,
    order_type: OrderType = OrderType.FOK,
):
    # Polymarket: maker amount max 2 decimals, taker amount max 4 decimals
    price = round(float(price), 2)
    size = int(float(size) * 10000) / 10000.0  # truncate to 4 decimals
    if size <= 0 or price <= 0:
        raise ValueError(f"Invalid order: price={price}, size={size}")
    order_args = OrderArgs(
        price=price,
        size=size,
        side=_side_to_constant(side),
        token_id=token_id,
    )
    signed_order = client.create_order(order_args)
    resp = client.post_order(signed_order, orderType=order_type)
    # post_order may return requests.Response or a dict depending on client version
    if hasattr(resp, "status_code"):
        if resp.status_code != 200:
            raise Exception(f"Failed to submit order: {getattr(resp, 'text', resp)}")
        return resp.json() if hasattr(resp, "json") and callable(resp.json) else resp
    if isinstance(resp, dict) and not resp.get("success", True):
        raise Exception(f"Failed to submit order: {resp.get('errorMsg', resp)}")
    return resp if isinstance(resp, dict) else {"success": True, "response": resp}


if __name__ == "__main__":
    # Test: create client and verify connection (no order placed).
    # Requires .env: PK, FUNDER, CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE
    def main():
        required = ["PK", "FUNDER", "CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE"]
        missing = [k for k in required if not (os.getenv(k) or "").strip()]
        if missing:
            print("Missing env vars:", ", ".join(missing))
            print("Set them in .env (see .env.example). Then run: python helpers/clob_client.py")
            return

        print("Creating CLOB client...")
        try:
            client = make_client()
        except ValueError as e:
            print("Error:", e)
            return

        print("  Host:", client.host)
        print("  Address:", client.get_address())

        print("\nTesting connection (get_ok)...")
        try:
            ok = client.get_ok()
            print("  OK:", ok)
        except Exception as e:
            print("  Failed:", e)
            return

        print("\nTesting server time...")
        try:
            t = client.get_server_time()
            print("  Server time:", t)
        except Exception as e:
            print("  Failed:", e)
            return

        print("\nAll checks passed. Client is ready (no order was placed).")

    main()