"""Verified public fee scenarios; never verification of the user's account tier."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config/wallex-official-fees.json"


def official_public_rates(path=DEFAULT_PATH, role="taker", level=1):
    """Return TMN/USDT bps for an explicit public tier; actual account is unknown."""
    if role not in {"maker", "taker"} or type(level) is not int or level < 1:
        raise ValueError("maker/taker role and positive integer level required")
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    units = document["units"]
    if units["fraction_to_percent"] != 100 or units["fraction_to_bps"] != 10000 or units["one_bps_fraction"] != .0001:
        raise ValueError("Official fee units differ from fractional rate/percent/bps contract")
    body = Path(document["source"]["raw_body_path"])
    if not body.is_absolute() and not body.exists():
        body = path.resolve().parents[1] / body
    raw_bytes = body.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != document["source"]["sha256"]:
        raise ValueError("Official fee source SHA256 mismatch")
    raw = json.loads(raw_bytes)
    if raw.get("success") is not True:
        raise ValueError("Official fee response did not report success")
    rates = {}
    for quote in ("TMN", "USDT"):
        try:
            row = document["spot_public_schedule"][quote]["tiers"][level - 1]
            evidence = raw["result"][quote]["levels"][level - 1]
        except IndexError as error:
            raise ValueError(f"No public level {level} for {quote}") from error
        fraction = Decimal(str(evidence[role + "_fee"]))
        if (not fraction.is_finite() or not 0 <= fraction < 1 or row["level"] != level or
                Decimal(str(row[role + "_fraction"])) != fraction or
                Decimal(str(row[role + "_percent"])) != fraction * 100 or
                Decimal(str(row[role + "_bps"])) != fraction * 10000):
            raise ValueError("Normalized public fee does not match original evidence and units")
        rates[quote] = float(fraction * 10000)
    return rates
