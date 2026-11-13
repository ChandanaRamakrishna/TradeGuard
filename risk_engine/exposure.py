from .io import load_positions

def portfolio_exposure(df=None):
    df = load_positions() if df is None else df.copy()
    if "position_value" not in df:
        df["position_value"] = df["quantity"] * df["current_price"]
    total = float(df["position_value"].sum())
    by_asset = df[["instrument", "position_value"]].to_dict("records")
    by_sector = (df.groupby("sector", as_index=False)["position_value"]
                   .sum().sort_values("position_value", ascending=False)
                   .to_dict("records"))
    return {"total_exposure": total, "by_asset": by_asset, "by_sector": by_sector}

def get_position(instrument, df=None):
    df = load_positions() if df is None else df.copy()
    matches = df[df["instrument"].str.casefold() == instrument.casefold()]
    if matches.empty:
        return {"error": f"No position found for {instrument}"}
    return matches.iloc[0].to_dict()
