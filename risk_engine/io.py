from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSITIONS_PATH = ROOT / "data" / "raw" / "trading_positions.csv"
RETURNS_PATH = ROOT / "data" / "processed" / "historical_returns.csv"

REQUIRED_COLUMNS = {
    "instrument", "quantity", "purchase_price", "current_price",
    "risk_limit", "sector", "desk"
}

def load_positions(path=POSITIONS_PATH):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    numeric = ["quantity", "purchase_price", "current_price", "risk_limit"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["quantity", "current_price", "risk_limit"]] < 0).any().any():
        raise ValueError("Quantity, current price and risk limit must be non-negative.")
    df["position_value"] = df["quantity"] * df["current_price"]
    return df
