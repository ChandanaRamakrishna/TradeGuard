import json
from pathlib import Path

from agent.graph import build_graph

ROOT = Path(__file__).resolve().parents[1]


def _path_value(data, dotted_path):
    value = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _evaluate_case(graph, case):
    result = graph.invoke(
        {"question": case["question"]},
        config={"configurable": {"thread_id": f"eval-{case['id']}"}},
    )
    actual_tools = [call["tool"] for call in result.get("tool_calls", [])]
    expected_tools = case["expected_tools"]
    tool_selection_passed = actual_tools == expected_tools

    numeric_checks = []
    for expected in case.get("expected_numeric", []):
        actual = _path_value(result, expected["path"])
        passed = (isinstance(actual, (int, float)) and
                  abs(actual - expected["value"]) <= expected.get("tolerance", 0))
        numeric_checks.append({**expected, "actual": actual, "passed": passed})

    required_outputs = case.get("required_outputs", [])
    missing_outputs = [key for key in required_outputs if result.get(key) is None]
    task_completion_passed = not missing_outputs

    evidence = result.get("evidence", {})
    successful_tools = {
        call["tool"] for call in evidence.get("tool_calls", []) if call.get("status") == "ok"
    }
    calculations = evidence.get("calculated", {})
    calculation_sources_valid = all(
        record.get("source") == "deterministic Python tool"
        and record.get("tool") in successful_tools
        for record in calculations.values()
    )
    policy_docs = evidence.get("policy", [])
    policy_sources_valid = all(
        doc.get("source") and doc.get("source") in evidence.get("policy_sources", [])
        for doc in policy_docs
    )
    if case.get("requires_policy_source"):
        policy_sources_valid = policy_sources_valid and bool(policy_docs)
    provenance_passed = calculation_sources_valid and policy_sources_valid

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "tool_selection_passed": tool_selection_passed,
        "numeric_checks": numeric_checks,
        "task_completion_passed": task_completion_passed,
        "missing_outputs": missing_outputs,
        "evidence_provenance_passed": provenance_passed,
        "policy_sources": evidence.get("policy_sources", []),
    }


def run_evaluation():
    cases = json.loads((ROOT / "evaluation" / "golden_dataset.json").read_text(encoding="utf-8"))
    graph = build_graph(enable_llm=False)
    results = [_evaluate_case(graph, case) for case in cases]

    tool_selection_accuracy = (
        sum(result["tool_selection_passed"] for result in results) / len(results) if results else 0
    )
    numeric_checks = [check for result in results for check in result["numeric_checks"]]
    numerical_accuracy = (
        sum(check["passed"] for check in numeric_checks) / len(numeric_checks)
        if numeric_checks else None
    )
    task_completion_rate = (
        sum(result["task_completion_passed"] for result in results) / len(results) if results else 0
    )
    provenance_rate = (
        sum(result["evidence_provenance_passed"] for result in results) / len(results)
        if results else 0
    )

    report = {
        "evaluation_type": "deterministic agent routing and evidence evaluation",
        "cases": len(results),
        "metrics": {
            "tool_selection_accuracy": {
                "score": tool_selection_accuracy,
                "passed": sum(row["tool_selection_passed"] for row in results),
                "total": len(results),
                "definition": "Exact ordered match of actual recorded tool calls to the golden sequence.",
            },
            "numerical_accuracy": {
                "score": numerical_accuracy,
                "passed": sum(check["passed"] for check in numeric_checks),
                "total": len(numeric_checks),
                "definition": "Expected deterministic outputs compared with per-value tolerances.",
            },
            "groundedness": {
                "evidence_provenance_coverage": provenance_rate,
                "passed": sum(row["evidence_provenance_passed"] for row in results),
                "total": len(results),
                "definition": "Structural check that calculation evidence names a successful deterministic tool and retrieved policy chunks have source attribution.",
                "limitation": "This is not semantic groundedness or an LLM hallucination evaluation.",
            },
            "task_completion": {
                "rate": task_completion_rate,
                "passed": sum(row["task_completion_passed"] for row in results),
                "total": len(results),
                "definition": "All requested output sections declared by each golden case are present.",
            },
        },
        "llm_evaluation": "Not performed; this evaluation disables LLM calls and does not claim hallucination testing.",
        "results": results,
    }
    (ROOT / "evaluation" / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"cases": report["cases"], "metrics": report["metrics"],
                      "llm_evaluation": report["llm_evaluation"]}, indent=2))
    return report


if __name__ == "__main__":
    run_evaluation()
