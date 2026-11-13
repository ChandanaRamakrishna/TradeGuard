import pandas as pd
from risk_engine.limit_detection import detect_limit_breaches
from risk_engine.exposure import portfolio_exposure
from risk_engine.concentration import concentration_analysis
from risk_engine.var_calculator import historical_var
from risk_engine.stress_testing import stress_test

def sample():
    return pd.DataFrame([
        {"instrument":"A","quantity":10,"current_price":100,"risk_limit":1100,"sector":"IT","desk":"D"},
        {"instrument":"B","quantity":10,"current_price":100,"risk_limit":100,"sector":"IT","desk":"D"},
    ])

def test_breach_statuses():
    rows = {x["instrument"]: x for x in detect_limit_breaches(sample())}
    assert rows["A"]["status"] == "Warning"
    assert rows["B"]["status"] == "Critical"

def test_exposure_total():
    assert portfolio_exposure(sample())["total_exposure"] == 2000.0

def test_sector_concentration():
    result = concentration_analysis(sample())
    assert result["sectors"][0]["share"] == 1.0
    assert len(result["flags"]) == 1

def test_var_rejects_invalid_confidence():
    import pytest
    with pytest.raises(ValueError, match="confidence"):
        historical_var(confidence=1.0)

def test_stress_test_rejects_non_finite_or_excessive_shock():
    import pytest
    with pytest.raises(ValueError, match="market_shock"):
        stress_test(sample(), market_shock=float("nan"))
    with pytest.raises(ValueError, match="market_shock"):
        stress_test(sample(), market_shock=1.5)

def test_limit_detection_rejects_invalid_threshold():
    import pytest
    with pytest.raises(ValueError, match="warning_threshold"):
        detect_limit_breaches(sample(), warning_threshold=0)
