import math
import pandas as pd
from .io import load_positions

def detect_limit_breaches(df=None, warning_threshold=0.90):
    if isinstance(warning_threshold, bool):
        raise ValueError("warning_threshold must be a finite number in (0, 1].")
    try:
        warning_threshold = float(warning_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("warning_threshold must be a finite number in (0, 1].") from exc
    if not math.isfinite(warning_threshold) or not 0 < warning_threshold <= 1:
        raise ValueError("warning_threshold must be a finite number in (0, 1].")
    data = load_positions() if df is None else df.copy()
    required = {"instrument", "quantity", "current_price", "risk_limit"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Positions missing required columns: {sorted(missing)}")
    base = data[["quantity", "current_price"]].apply(pd.to_numeric, errors="raise")
    if not base.map(math.isfinite).all().all():
        raise ValueError("Quantities and current prices must be finite.")
    if (base < 0).any().any():
        raise ValueError("Quantities and current prices must be non-negative.")
    data[["quantity", "current_price"]] = base
    if "position_value" not in data:
        data["position_value"] = data["quantity"] * data["current_price"]
    numeric = data[["position_value", "risk_limit"]].apply(pd.to_numeric, errors="raise")
    if not numeric.map(math.isfinite).all().all():
        raise ValueError("Position values and risk limits must be finite.")
    if (numeric < 0).any().any():
        raise ValueError("Position values and risk limits must be non-negative.")
    data[["position_value", "risk_limit"]] = numeric
    results = []
    for row in data.to_dict("records"):
        limit = float(row["risk_limit"])
        value = float(row["position_value"])
        utilization = value / limit if limit > 0 else (float("inf") if value > 0 else 0.0)
        if utilization > 1.0:
            status = "Critical"
        elif utilization >= warning_threshold:
            status = "Warning"
        else:
            status = "Normal"
        results.append({
            "instrument": row["instrument"], "position_value": value,
            "risk_limit": limit, "excess": max(value-limit, 0.0),
            "utilization": utilization, "breach_pct": max(utilization-1.0, 0.0)*100,
            "status": status, "sector": row.get("sector"), "desk": row.get("desk")
        })
    return sorted(results, key=lambda x: x["utilization"], reverse=True)
