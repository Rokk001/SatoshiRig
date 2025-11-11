"""Formatting utilities for hash values and time."""


def format_hash_number(value: float, unit: str = "H/s") -> str:
    """Format hash numbers with magnitude units (K, M, G, T, P, E)"""
    if value == 0:
        return f"0 {unit}"
    
    abs_value = abs(value)
    if abs_value < 1000:
        return f"{value:.2f} {unit}"
    elif abs_value < 1_000_000:
        return f"{value / 1000:.2f} K{unit}"
    elif abs_value < 1_000_000_000:
        return f"{value / 1_000_000:.2f} M{unit}"
    elif abs_value < 1_000_000_000_000:
        return f"{value / 1_000_000_000:.2f} G{unit}"
    elif abs_value < 1_000_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} T{unit}"
    elif abs_value < 1_000_000_000_000_000_000:
        return f"{value / 1_000_000_000_000_000:.2f} P{unit}"
    else:
        return f"{value / 1_000_000_000_000_000_000:.2f} E{unit}"


def format_time_to_block(seconds: float) -> str:
    """Convert seconds to human-readable format: years, months, days"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        # Convert to years, months, days
        total_days = seconds / 86400
        years = int(total_days / 365)
        remaining_days = total_days - (years * 365)
        months = int(remaining_days / 30)
        days = remaining_days - (months * 30)
        
        parts = []
        if years > 0:
            parts.append(f"{years} {'year' if years == 1 else 'years'}")
        if months > 0:
            parts.append(f"{months} {'month' if months == 1 else 'months'}")
        if days > 0 or len(parts) == 0:
            parts.append(f"{days:.1f} {'day' if days == 1 else 'days'}")
        
        return ", ".join(parts)

