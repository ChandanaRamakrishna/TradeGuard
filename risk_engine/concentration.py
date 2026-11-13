from .exposure import portfolio_exposure

def concentration_analysis(df=None, threshold=0.40):
    exposure = portfolio_exposure(df)
    total = exposure["total_exposure"]
    sectors = []
    for row in exposure["by_sector"]:
        share = (row["position_value"] / total) if total else 0.0
        sectors.append({
            "sector": row["sector"],
            "exposure": float(row["position_value"]),
            "share": float(share),
            "flag": bool(share > threshold),
        })
    return {"threshold": threshold, "sectors": sectors,
            "flags": [x for x in sectors if x["flag"]]}
