from agent.tools import lookup_position, check_limit_breaches

def test_lookup_infosys():
    result = lookup_position.invoke({"instrument":"Infosys"})
    assert result["position_value"] == 1800000
    assert result["risk_limit"] == 1500000

def test_breach_tool_marks_infosys_critical():
    rows = check_limit_breaches.invoke({})
    infosys = next(x for x in rows if x["instrument"] == "Infosys")
    assert infosys["status"] == "Critical"
