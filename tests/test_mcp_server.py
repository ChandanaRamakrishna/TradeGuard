from mcp_server.server import (
    check_portfolio_concentration,
    run_portfolio_stress_test,
)


def test_mcp_concentration_tool_returns_deterministic_flags():
    result = check_portfolio_concentration()
    assert result["threshold"] == 0.40
    assert "sectors" in result
    assert "flags" in result


def test_mcp_stress_tool_returns_deterministic_portfolio_pnl():
    result = run_portfolio_stress_test(market_shock=-0.05)
    assert result["estimated_portfolio_pnl"] < 0
    assert len(result["positions"]) > 0
