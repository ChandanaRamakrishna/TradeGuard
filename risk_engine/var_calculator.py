from pathlib import Path
import math
import numpy as np
import pandas as pd
from .io import ROOT, RETURNS_PATH, load_positions

def generate_synthetic_returns(path=RETURNS_PATH, days=260, seed=17):
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Synthetic daily returns with modest volatility; deliberately not real market data.
    instruments = ["Reliance", "Infosys", "HDFC Bank", "TCS", "ICICI Bank", "Wipro", "Tata Motors"]
    common = rng.normal(0.0002, 0.009, days)
    data = {"date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)}
    for i, name in enumerate(instruments):
        idio = rng.normal(0.0, 0.008 + i*0.0004, days)
        data[name] = np.clip(common * 0.45 + idio, -0.12, 0.12)
    pd.DataFrame(data).to_csv(path, index=False)
    return path

def historical_var(df=None, returns_path=RETURNS_PATH, confidence=0.95):
    if isinstance(confidence, bool):
        raise ValueError("confidence must be a number strictly between 0 and 1.")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number strictly between 0 and 1.") from exc
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be a finite number strictly between 0 and 1.")

    positions = load_positions() if df is None else df.copy()
    required = {"instrument", "quantity", "current_price"}
    missing_columns = required - set(positions.columns)
    if missing_columns:
        raise ValueError(f"Positions missing required columns: {sorted(missing_columns)}")
    if positions.empty:
        raise ValueError("Cannot calculate VaR for an empty portfolio.")
    try:
        base = positions[["quantity", "current_price"]].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Quantities and current prices must be numeric.") from exc
    if not base.map(math.isfinite).all().all() or (base < 0).any().any():
        raise ValueError("Quantities and current prices must be finite and non-negative.")
    positions[["quantity", "current_price"]] = base
    if not Path(returns_path).exists():
        generate_synthetic_returns(returns_path)
    returns = pd.read_csv(returns_path)
    if returns.empty:
        raise ValueError("Returns data is empty; cannot calculate VaR.")
    pnl = np.zeros(len(returns), dtype=float)
    missing = []
    for row in positions.to_dict("records"):
        name = row["instrument"]
        if name not in returns.columns:
            missing.append(name)
            continue
        try:
            value = float(row.get("position_value", row["quantity"] * row["current_price"]))
            instrument_returns = pd.to_numeric(returns[name], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Position or returns for {name} must be numeric.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Position value for {name} must be finite and non-negative.")
        if not np.isfinite(instrument_returns).all():
            raise ValueError(f"Returns for {name} contain non-finite values.")
        pnl += instrument_returns * value
    if missing:
        raise ValueError(f"Returns missing for: {', '.join(missing)}")
    q = float(np.quantile(pnl, 1-confidence))
    return {
        "method": "historical simulation on synthetic daily returns",
        "confidence": confidence, "horizon": "1 trading day",
        "var": max(0.0, -q), "pnl_5th_percentile": q,
        "observations": int(len(pnl)),
        "warning": "Synthetic demonstration estimate; not a production risk measure."
    }
