from langchain_core.tools import tool
from risk_engine.exposure import portfolio_exposure, get_position
from risk_engine.limit_detection import detect_limit_breaches
from risk_engine.var_calculator import historical_var
from risk_engine.concentration import concentration_analysis
from risk_engine.stress_testing import stress_test

@tool
def lookup_position(instrument: str) -> dict:
    """Retrieve a simulated trading position by instrument name."""
    return get_position(instrument)

@tool
def calculate_exposure() -> dict:
    """Calculate total portfolio, asset and sector exposure."""
    return portfolio_exposure()

@tool
def check_limit_breaches() -> list:
    """Run deterministic position limit checks for every instrument."""
    return detect_limit_breaches()

@tool
def calculate_historical_var(confidence: float = 0.95) -> dict:
    """Calculate one-day historical VaR from synthetic daily returns."""
    return historical_var(confidence=confidence)

@tool
def check_concentration() -> dict:
    """Calculate sector concentration and flag shares above the sample threshold."""
    return concentration_analysis()

@tool
def run_stress_test(market_shock: float = -0.05, sector_shocks: dict = None) -> dict:
    """Estimate deterministic portfolio P&L under parallel and optional sector shocks."""
    return stress_test(market_shock=market_shock, sector_shocks=sector_shocks)
