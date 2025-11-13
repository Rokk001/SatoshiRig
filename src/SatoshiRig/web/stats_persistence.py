"""Persistent statistics storage for SatoshiRig."""
import json
import os
from datetime import datetime
from typing import Dict

from ..db import get_value, set_value

# Backward-compatibility: legacy JSON file path
DEFAULT_STATS_FILE = os.path.join(
    os.environ.get("DATA_DIR", "/app/data"),
    "statistics.json"
)


def _load_db_stats() -> Dict:
    stats = {}
    stats["total_hashes"] = int(float(get_value("stats", "total_hashes", "0") or 0))
    stats["peak_hash_rate"] = float(get_value("stats", "peak_hash_rate", "0.0") or 0.0)
    stats["shares_submitted"] = int(get_value("stats", "shares_submitted", "0") or 0)
    stats["shares_accepted"] = int(get_value("stats", "shares_accepted", "0") or 0)
    stats["shares_rejected"] = int(get_value("stats", "shares_rejected", "0") or 0)
    stats["last_updated"] = get_value("stats", "last_updated")
    return stats


def load_statistics() -> Dict:
    """Load persistent statistics from storage."""
    stats = _load_db_stats()

    # If DB is empty but legacy JSON exists, migrate once
    if (
        stats["total_hashes"] == 0
        and stats["peak_hash_rate"] == 0.0
        and stats["shares_submitted"] == 0
        and os.path.exists(DEFAULT_STATS_FILE)
    ):
        try:
            with open(DEFAULT_STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats["total_hashes"] = float(data.get("total_hashes", 0))
            stats["peak_hash_rate"] = float(data.get("peak_hash_rate", 0.0))
            stats["shares_submitted"] = int(data.get("shares_submitted", 0))
            stats["shares_accepted"] = int(data.get("shares_accepted", 0))
            stats["shares_rejected"] = int(data.get("shares_rejected", 0))
            stats["last_updated"] = data.get("last_updated")
            # Persist migrated data to DB
            save_statistics(
                stats["total_hashes"],
                stats["peak_hash_rate"],
                stats["shares_submitted"],
                stats["shares_accepted"],
                stats["shares_rejected"],
            )
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    return stats


def save_statistics(
    total_hashes: float,
    peak_hash_rate: float,
    shares_submitted: int,
    shares_accepted: int,
    shares_rejected: int
):
    """Save statistics to persistent storage."""
    set_value("stats", "total_hashes", str(total_hashes))
    set_value("stats", "peak_hash_rate", str(peak_hash_rate))
    set_value("stats", "shares_submitted", str(shares_submitted))
    set_value("stats", "shares_accepted", str(shares_accepted))
    set_value("stats", "shares_rejected", str(shares_rejected))
    set_value("stats", "last_updated", datetime.utcnow().isoformat())


def merge_statistics(
    existing_total_hashes: int,
    existing_peak_hash_rate: float,
    existing_shares_submitted: int,
    existing_shares_accepted: int,
    existing_shares_rejected: int,
    new_total_hashes: int,
    new_peak_hash_rate: float,
    new_shares_submitted: int,
    new_shares_accepted: int,
    new_shares_rejected: int
) -> Dict:
    """
    Merge new statistics with existing persistent statistics.
    Returns merged statistics.
    """
    # For hashes and shares, add new values to existing
    merged_total_hashes = existing_total_hashes + new_total_hashes
    merged_shares_submitted = existing_shares_submitted + new_shares_submitted
    merged_shares_accepted = existing_shares_accepted + new_shares_accepted
    merged_shares_rejected = existing_shares_rejected + new_shares_rejected

    # For peak hash rate, take the maximum
    merged_peak_hash_rate = max(existing_peak_hash_rate, new_peak_hash_rate)

    return {
        "total_hashes": merged_total_hashes,
        "peak_hash_rate": merged_peak_hash_rate,
        "shares_submitted": merged_shares_submitted,
        "shares_accepted": merged_shares_accepted,
        "shares_rejected": merged_shares_rejected
    }

