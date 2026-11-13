import math
import pandas as pd
from .io import load_positions

def stress_test(df=None, market_shock=-0.05, sector_shocks=None):
    def validate_shock(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite shock between -1.0 and 1.0.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite shock between -1.0 and 1.0.") from exc
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite shock between -1.0 and 1.0.")
        return value

    market_shock = validate_shock(market_shock, "market_shock")
    if sector_shocks is None:
        sector_shocks = {}
    if not isinstance(sector_shocks, dict):
        raise ValueError("sector_shocks must be a mapping of sector names to shocks.")
    if any(not isinstance(sector, str) or not sector.strip() for sector in sector_shocks):
        raise ValueError("sector_shocks keys must be non-empty sector names.")
    sector_shocks = {sector: validate_shock(shock, f"sector shock for {sector}")
                     for sector, shock in sector_shocks.items()}
    data = load_positions() if df is None else df.copy()
    required = {"instrument", "quantity", "current_price", "sector"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Positions missing required columns: {sorted(missing)}")
    try:
        base = data[["quantity", "current_price"]].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Quantities and current prices must be numeric.") from exc
    if not base.map(math.isfinite).all().all() or (base < 0).any().any():
        raise ValueError("Quantities and current prices must be finite and non-negative.")
    data[["quantity", "current_price"]] = base
    if "position_value" not in data:
        data["position_value"] = data["quantity"] * data["current_price"]
    try:
        values = pd.to_numeric(data["position_value"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Position values must be numeric.") from exc
    if not values.map(math.isfinite).all() or (values < 0).any():
        raise ValueError("Position values must be finite and non-negative.")
    data["position_value"] = values
    unknown_sectors = set(sector_shocks) - set(data["sector"].dropna().unique())
    if unknown_sectors:
        raise ValueError(f"Unknown sectors in sector_shocks: {sorted(unknown_sectors)}")
    rows = []
    total_pnl = 0.0
    for row in data.to_dict("records"):
        shock = float(sector_shocks.get(row["sector"], market_shock))
        pnl = float(row["position_value"]) * shock
        total_pnl += pnl
        rows.append({"instrument": row["instrument"], "sector": row["sector"],
                     "shock_pct": shock*100, "estimated_pnl": pnl})
    return {"scenario": "user-defined deterministic shock",
            "estimated_portfolio_pnl": total_pnl, "positions": rows}
