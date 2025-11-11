"""Status management for the web dashboard."""
import threading
from collections import deque
from datetime import datetime
from typing import Dict

STATUS_LOCK = threading.Lock()
STATUS: Dict = {
    "running": False,
    "current_height": 0,
    "best_difficulty": 0.0,
    "hash_rate": 0.0,
    "last_hash": None,
    "uptime_seconds": 0,
    "start_time": None,
    "wallet_address": None,
    "explorer_url": None,
    "pool_connected": False,
    "pool_host": None,
    "pool_port": None,
    "job_id": None,
    "total_hashes": 0,
    "peak_hash_rate": 0.0,
    "average_hash_rate": 0.0,
    "shares_submitted": 0,
    "shares_accepted": 0,
    "shares_rejected": 0,
    "last_share_time": None,
    "hash_rate_history": deque(maxlen=60),  # Last 60 data points (2 minutes at 2s intervals)
    "difficulty_history": deque(maxlen=60),
    # Performance & Monitoring (Feature 1)
    "cpu_usage": 0.0,
    "memory_usage": 0.0,
    "gpu_usage": 0.0,
    "gpu_temperature": 0.0,
    "gpu_memory": 0.0,
    # Mining Intelligence (Feature 2)
    "estimated_time_to_block": None,
    "block_found_probability": 0.0,
    "estimated_profitability": 0.0,
    "difficulty_trend": "stable",  # increasing, decreasing, stable
    "network_difficulty": 0.0,
    "target_difficulty": 0.0,
    "errors": []
}

# Statistics tracking
STATS_LOCK = threading.Lock()
STATS = {
    "total_hashes": 0,
    "peak_hash_rate": 0.0,
    "hash_rate_samples": deque(maxlen=300),  # Last 300 samples for average
    "shares": deque(maxlen=100),  # Last 100 shares to prevent memory leak
    "start_time": None
}


def update_status(key: str, value):
    """Update a status value (thread-safe)"""
    with STATUS_LOCK:
        STATUS[key] = value
        # Update statistics
        if key == "hash_rate" and value:
            with STATS_LOCK:
                STATS["hash_rate_samples"].append(value)
                if value > STATS["peak_hash_rate"]:
                    STATS["peak_hash_rate"] = value
                    STATUS["peak_hash_rate"] = value
                if len(STATS["hash_rate_samples"]) > 0:
                    avg = sum(STATS["hash_rate_samples"]) / len(STATS["hash_rate_samples"])
                    STATUS["average_hash_rate"] = avg
        if key == "total_hashes":
            with STATS_LOCK:
                STATS["total_hashes"] = value
        if key == "shares_submitted":
            with STATS_LOCK:
                STATS["shares"].append({
                    "timestamp": datetime.now().isoformat(),
                    "accepted": value
                })


def get_status() -> Dict:
    """Get current status (thread-safe copy)"""
    with STATUS_LOCK:
        status = STATUS.copy()
        # Add history arrays
        status["hash_rate_history"] = list(status["hash_rate_history"])
        status["difficulty_history"] = list(status["difficulty_history"])
        # Add statistics
        with STATS_LOCK:
            status["total_hashes"] = STATS["total_hashes"]
            # Convert deque to list and get last 10 shares
            shares_list = list(STATS["shares"])
            status["shares"] = shares_list[-10:]  # Last 10 shares
        return status


def add_share(accepted: bool, response: str = None):
    """Add a share to statistics"""
    with STATUS_LOCK:
        if accepted:
            STATUS["shares_accepted"] += 1
        else:
            STATUS["shares_rejected"] += 1
        STATUS["shares_submitted"] += 1
        STATUS["last_share_time"] = datetime.now().isoformat()
        update_status("shares_submitted", STATUS["shares_submitted"])


def update_pool_status(connected: bool, host: str = None, port: int = None):
    """Update pool connection status"""
    with STATUS_LOCK:
        STATUS["pool_connected"] = connected
        if host:
            STATUS["pool_host"] = host
        if port:
            STATUS["pool_port"] = port

