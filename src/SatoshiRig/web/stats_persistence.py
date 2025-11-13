"""Persistent statistics storage for SatoshiRig."""
import json
import os
import threading
from datetime import datetime
from typing import Dict


# Default path for statistics file
DEFAULT_STATS_FILE = os.path.join(
    os.environ.get("DATA_DIR", "/app/data"),
    "statistics.json"
)

# Lock for thread-safe file operations
FILE_LOCK = threading.Lock()


def ensure_data_dir():
    """Ensure the data directory exists."""
    stats_file = os.environ.get("STATS_FILE", DEFAULT_STATS_FILE)
    data_dir = os.path.dirname(os.path.abspath(stats_file))
    os.makedirs(data_dir, exist_ok=True)


def load_statistics() -> Dict:
    """Load persistent statistics from file."""
    stats_file = os.environ.get("STATS_FILE", DEFAULT_STATS_FILE)
    
    if not os.path.exists(stats_file):
        return {
            "total_hashes": 0,
            "peak_hash_rate": 0.0,
            "shares_submitted": 0,
            "shares_accepted": 0,
            "shares_rejected": 0,
            "last_updated": None
        }
    
    try:
        with FILE_LOCK:
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all required keys exist
                stats = {
                    "total_hashes": data.get("total_hashes", 0),
                    "peak_hash_rate": data.get("peak_hash_rate", 0.0),
                    "shares_submitted": data.get("shares_submitted", 0),
                    "shares_accepted": data.get("shares_accepted", 0),
                    "shares_rejected": data.get("shares_rejected", 0),
                    "last_updated": data.get("last_updated")
                }
                return stats
    except (json.JSONDecodeError, IOError, OSError) as e:
        # If file is corrupted or can't be read, return defaults
        import logging
        logger = logging.getLogger("SatoshiRig.stats")
        logger.warning(f"Failed to load statistics from {stats_file}: {e}. Using defaults.")
        return {
            "total_hashes": 0,
            "peak_hash_rate": 0.0,
            "shares_submitted": 0,
            "shares_accepted": 0,
            "shares_rejected": 0,
            "last_updated": None
        }


def save_statistics(
    total_hashes: int,
    peak_hash_rate: float,
    shares_submitted: int,
    shares_accepted: int,
    shares_rejected: int
):
    """Save statistics to persistent storage."""
    stats_file = os.environ.get("STATS_FILE", DEFAULT_STATS_FILE)
    ensure_data_dir()
    
    stats = {
        "total_hashes": total_hashes,
        "peak_hash_rate": peak_hash_rate,
        "shares_submitted": shares_submitted,
        "shares_accepted": shares_accepted,
        "shares_rejected": shares_rejected,
        "last_updated": datetime.now().isoformat()
    }
    
    try:
        with FILE_LOCK:
            # Write to temporary file first, then rename (atomic operation)
            temp_file = stats_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            # Atomic rename
            if os.path.exists(temp_file):
                if os.path.exists(stats_file):
                    os.replace(temp_file, stats_file)
                else:
                    os.rename(temp_file, stats_file)
    except (IOError, OSError) as e:
        import logging
        logger = logging.getLogger("SatoshiRig.stats")
        logger.error(f"Failed to save statistics to {stats_file}: {e}")


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

