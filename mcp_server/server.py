from mcp.server.fastmcp import FastMCP
from risk_engine.limit_detection import detect_limit_breaches
from risk_engine.var_calculator import historical_var
from risk_engine.exposure import portfolio_exposure
from risk_engine.concentration import concentration_analysis
from risk_engine.stress_testing import stress_test

mcp = FastMCP("TradeGuard Risk Tools")

@mcp.tool()
def get_portfolio_exposure() -> dict:
    """Return deterministic portfolio and sector exposure calculations."""
    return portfolio_exposure()

@mcp.tool()
def detect_trading_limit_breaches() -> list:
    """Return deterministic Normal, Warning and Critical limit classifications."""
    return detect_limit_breaches()

@mcp.tool()
def calculate_portfolio_var(confidence: float = 0.95) -> dict:
    """Calculate one-day historical VaR using synthetic returns."""
    return historical_var(confidence=confidence)

@mcp.tool()
def check_portfolio_concentration(threshold: float = 0.40) -> dict:
    """Calculate sector concentration and flag shares above the threshold."""
    return concentration_analysis(threshold=threshold)

@mcp.tool()
def run_portfolio_stress_test(market_shock: float = -0.05,
                              sector_shocks: dict = None) -> dict:
    """Estimate portfolio P&L for a parallel shock and optional sector shocks."""
    return stress_test(market_shock=market_shock, sector_shocks=sector_shocks)

if __name__ == "__main__":
    mcp.run()
