import json
import logging
import os
import re
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.rag import retrieve_policy
from agent.tools import (
    calculate_exposure,
    calculate_historical_var,
    check_concentration,
    check_limit_breaches,
    lookup_position,
    run_stress_test,
)
from risk_engine.io import ROOT

load_dotenv(ROOT / ".env")
logger = logging.getLogger(__name__)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

INSTRUMENTS = (
    "Reliance", "Infosys", "HDFC Bank", "TCS", "ICICI Bank", "Wipro", "Tata Motors"
)
SECTORS = ("Energy", "IT", "Banking", "Automobile")
FINANCIAL_TOOLS = {
    "lookup_position": lookup_position,
    "calculate_exposure": calculate_exposure,
    "check_limit_breaches": check_limit_breaches,
    "calculate_historical_var": calculate_historical_var,
    "check_concentration": check_concentration,
    "run_stress_test": run_stress_test,
}


class InvestigationState(TypedDict, total=False):
    question: str
    instrument: str
    selected_tools: list[str]
    tool_calls: list[dict]
    position: dict
    breach: dict
    breaches: list[dict]
    exposure: dict
    var: dict
    concentration: dict
    stress: dict
    policy: list[dict]
    policy_error: str
    evidence: dict
    explanation: str
    recommendation: str
    approval: dict
    status: str
    llm_status: str


def _instrument_from_question(question: str) -> str:
    for name in INSTRUMENTS:
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE):
            return name
    return ""


def plan_tools(question: str, instrument: str = "") -> list[str]:
    """Select a minimal deterministic tool set from the requested information."""
    q = question.casefold()
    instrument = instrument or _instrument_from_question(question)
    wants_portfolio = any(term in q for term in (
        "portfolio", "overall", "all positions", "whole book", "entire book"
    ))
    wants_exposure = any(term in q for term in (
        "exposure", "portfolio value", "total value"
    )) or ("market value" in q and not instrument)
    wants_limits = any(term in q for term in (
        "limit", "breach", "alert", "utilization", "critical", "warning",
        "normal", "risk status", "risk level", "risk position", "why"
    ))
    wants_var = bool(re.search(r"\bvar\b", q)) or "value at risk" in q
    wants_concentration = any(term in q for term in ("concentration", "concentrated"))
    wants_sector_exposure = "sector" in q and wants_exposure
    wants_stress = any(term in q for term in (
        "stress", "shock", "scenario", "stressed", "drop", "decline", "fall"
    ))
    wants_policy = any(term in q for term in (
        "policy", "escalat", "analyst", "recommend", "rule"
    )) or ("why" in q and wants_limits)
    if "threshold" in q and not wants_concentration and not wants_limits:
        wants_policy = True

    selected = []
    if instrument:
        # Position lookup is relevant to any instrument-specific request.
        selected.append("lookup_position")
        if wants_limits:
            selected.append("check_limit_breaches")
    elif wants_portfolio and not any((wants_exposure, wants_limits, wants_var,
                                      wants_concentration, wants_stress)):
        # A broad portfolio-risk question gets a compact risk snapshot.
        selected.extend(("calculate_exposure", "check_limit_breaches",
                         "calculate_historical_var"))

    if wants_exposure or wants_sector_exposure:
        selected.append("calculate_exposure")
    policy_only_question = wants_policy and not instrument and not any(term in q for term in (
        "positions are", "over their", "my portfolio", "current utilization", "calculate"
    ))
    if wants_limits and not instrument and not policy_only_question:
        selected.append("check_limit_breaches")
    if wants_var:
        selected.append("calculate_historical_var")
    if wants_concentration:
        selected.append("check_concentration")
    if wants_stress:
        selected.append("run_stress_test")
    if wants_policy:
        selected.append("retrieve_policy")

    # Preserve order and remove duplicates when several intents overlap.
    return list(dict.fromkeys(selected))


def _stress_arguments(question: str) -> dict:
    sector_pattern = re.compile(
        r"\b(Energy|IT|Banking|Automobile)\b\s*(?:sector\s*)?"
        r"(?:(up|down|rise|fall|gain|loss|at|to|:)\s*)?"
        r"([+-]?\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    sector_shocks = {}
    sector_spans = []
    for match in sector_pattern.finditer(question):
        sector, direction, raw_value = match.groups()
        value = float(raw_value) / 100
        direction = (direction or "").casefold()
        if direction in {"down", "fall", "loss"}:
            value = -abs(value)
        elif direction in {"up", "rise", "gain"}:
            value = abs(value)
        elif not raw_value.startswith(("+", "-")):
            value = -abs(value)
        sector_shocks[next(name for name in SECTORS if name.casefold() == sector.casefold())] = value
        sector_spans.append(match.span(3))

    percentages = list(re.finditer(r"[+-]?\d+(?:\.\d+)?\s*%", question))
    percentages = [match for match in percentages
                   if not any(start <= match.start() < end for start, end in sector_spans)]
    shock_context = [match for match in percentages
                     if any(term in question[max(0, match.start() - 24):match.end() + 24].casefold()
                            for term in ("market", "parallel", "portfolio", "shock", "drop", "fall", "decline"))]
    candidates = shock_context or (percentages if not sector_shocks else [])
    market_shock = -0.05
    if candidates:
        match = candidates[0]
        raw = match.group().replace("%", "").strip()
        market_shock = float(raw) / 100
        context = question[max(0, match.start() - 24):match.end() + 24].casefold()
        if raw.startswith("+"):
            market_shock = abs(market_shock)
        elif raw.startswith("-") or any(term in context for term in ("drop", "fall", "decline", "down", "loss")):
            market_shock = -abs(market_shock)
        elif any(term in context for term in ("rise", "up", "gain", "rally")):
            market_shock = abs(market_shock)
        else:
            market_shock = -abs(market_shock)
    return {"market_shock": market_shock, "sector_shocks": sector_shocks}


def _tool_arguments(tool_name: str, state: InvestigationState) -> dict:
    if tool_name == "lookup_position":
        return {"instrument": state["instrument"]}
    if tool_name == "calculate_historical_var":
        return {"confidence": 0.95}
    if tool_name == "run_stress_test":
        return _stress_arguments(state["question"])
    return {}


def _plan(state: InvestigationState):
    question = state.get("question", "")
    instrument = state.get("instrument") or _instrument_from_question(question)
    selected = plan_tools(question, instrument)
    if not selected:
        return {
            "instrument": instrument,
            "selected_tools": [],
            "status": "needs_clarification",
            "explanation": ("Please specify an instrument or ask a portfolio-level question, "
                            "such as total exposure or portfolio VaR."),
        }
    return {"instrument": instrument, "selected_tools": selected, "status": "planned"}


def _execute(state: InvestigationState):
    outputs: dict[str, Any] = {}
    calls = []
    for tool_name in state.get("selected_tools", []):
        args = _tool_arguments(tool_name, state)
        try:
            if tool_name == "retrieve_policy":
                result = retrieve_policy(state["question"])
                outputs["policy"] = result
            else:
                result = FINANCIAL_TOOLS[tool_name].invoke(args)
                if tool_name == "lookup_position":
                    outputs["position"] = result
                elif tool_name == "calculate_exposure":
                    outputs["exposure"] = result
                elif tool_name == "check_limit_breaches":
                    outputs["breaches"] = result
                    outputs["breach"] = next(
                        (row for row in result if row["instrument"] == state.get("instrument")), {}
                    ) if state.get("instrument") else {}
                elif tool_name == "calculate_historical_var":
                    outputs["var"] = result
                elif tool_name == "check_concentration":
                    outputs["concentration"] = result
                elif tool_name == "run_stress_test":
                    outputs["stress"] = result
            calls.append({"tool": tool_name, "arguments": args, "status": "ok"})
        except Exception as exc:
            calls.append({"tool": tool_name, "arguments": args,
                          "status": "error", "error": type(exc).__name__})
            if tool_name == "retrieve_policy":
                outputs["policy_error"] = f"Policy retrieval failed ({type(exc).__name__})."
            outputs.setdefault("tool_errors", []).append(
                {"tool": tool_name, "error": type(exc).__name__, "message": str(exc)}
            )

    evidence = {
        "calculated": {
            key: {"source": "deterministic Python tool", "tool": tool_name, "result": outputs[key]}
            for key, tool_name in (
                ("position", "lookup_position"), ("breach", "check_limit_breaches"),
                ("breaches", "check_limit_breaches"), ("exposure", "calculate_exposure"),
                ("var", "calculate_historical_var"), ("concentration", "check_concentration"),
                ("stress", "run_stress_test"),
            ) if key in outputs
        },
        "policy": outputs.get("policy", []),
        "policy_sources": list(dict.fromkeys(
            doc.get("source", "unknown") for doc in outputs.get("policy", [])
        )),
        "tool_calls": calls,
    }
    return {**outputs, "tool_calls": calls, "evidence": evidence,
            "status": "investigated"}


def _deterministic_explanation(state: InvestigationState) -> str:
    lines = ["Calculated financial results (from deterministic Python tools):"]
    position = state.get("position", {})
    breach = state.get("breach", {})
    if position and "error" not in position:
        lines.append(
            f"- {state.get('instrument')}: market value {position.get('position_value', 'unavailable')}; "
            f"configured limit {position.get('risk_limit', 'unavailable')}."
        )
    if breach:
        lines.append(
            f"- Limit utilization {breach['utilization']:.1%}; classification {breach['status']}."
        )
    if state.get("exposure"):
        lines.append(f"- Total portfolio exposure: {state['exposure']['total_exposure']:,.2f}.")
    if state.get("var"):
        lines.append(
            f"- Portfolio {state['var']['confidence']:.0%} one-day synthetic VaR: "
            f"{state['var']['var']:,.2f}."
        )
    if state.get("concentration"):
        flags = state["concentration"].get("flags", [])
        lines.append(f"- Concentration flags: {len(flags)} sector(s) exceed the configured threshold.")
    if state.get("stress"):
        lines.append(
            f"- Estimated portfolio P&L under the selected stress: "
            f"{state['stress']['estimated_portfolio_pnl']:,.2f}."
        )
    if state.get("breaches") and not state.get("instrument"):
        critical = [row["instrument"] for row in state["breaches"] if row["status"] == "Critical"]
        lines.append(f"- Critical positions: {', '.join(critical) if critical else 'none'}.")

    policy = state.get("policy", [])
    lines.append("\nRetrieved policy facts (separate from calculations):")
    if policy:
        for doc in policy:
            lines.append(f"- {doc['text'].strip()} [Source: {doc.get('source', 'unknown')}]")
    else:
        lines.append("- No policy text was retrieved for this request.")
    if state.get("policy_error"):
        lines.append(f"- Policy retrieval issue: {state['policy_error']}")
    if state.get("tool_errors"):
        lines.append("- One or more requested tools failed; see the recorded tool-call evidence.")
    return "\n".join(lines)


def _groq_failure_category(exc: Exception) -> tuple[str, Optional[int]]:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    name = type(exc).__name__.casefold()
    if status == 401:
        category = "authentication_failed"
    elif status == 403:
        category = "permission_denied"
    elif status == 404:
        category = "model_not_found_or_unavailable"
    elif status == 429:
        category = "rate_limit_or_quota"
    elif status is not None and status >= 500:
        category = "provider_server_error"
    elif "timeout" in name or "connection" in name or "connecterror" in name:
        category = "connectivity_or_timeout"
    elif status == 400:
        category = "invalid_request_or_model_configuration"
    elif name == "emptyllmresponse":
        category = "empty_response"
    else:
        category = "provider_client_error"
    return category, status


def _explain(state: InvestigationState, enable_llm: bool = True):
    fallback = _deterministic_explanation(state)
    api_key = os.getenv("GROQ_API_KEY")
    llm_status = "disabled"
    if not enable_llm:
        explanation = fallback
    elif not api_key:
        llm_status = "missing_api_key"
        explanation = fallback + "\nLLM explanation skipped (missing_api_key); deterministic explanation used."
    else:
        try:
            model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            llm = ChatGroq(model=model, temperature=0)
            evidence = json.dumps(state.get("evidence", {}), ensure_ascii=False, default=str)
            msg = llm.invoke([
                SystemMessage(content=(
                    "Explain the request using only the supplied evidence. Clearly separate "
                    "calculated financial results from retrieved policy facts, cite each policy "
                    "source, and do not invent values or rules. Calculations are authoritative "
                    "deterministic Python outputs. State that this is a prototype. Never approve "
                    "an escalation or claim a trade was blocked."
                )),
                HumanMessage(content=f"Question: {state.get('question', '')}\nEvidence JSON: {evidence}"),
            ])
            explanation = str(msg.content).strip()
            if not explanation:
                raise RuntimeError("Groq returned an empty explanation.")
            llm_status = "success"
        except Exception as exc:
            category, status = _groq_failure_category(exc)
            logger.warning(
                "Groq explanation failed: category=%s exception=%s http_status=%s model=%s; "
                "using deterministic fallback.",
                category, type(exc).__name__, status, os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            )
            llm_status = category
            explanation = fallback + f"\nLLM explanation unavailable ({category}); deterministic explanation used."

    breach = state.get("breach", {})
    critical = breach.get("status") == "Critical" or any(
        row.get("status") == "Critical" for row in state.get("breaches", [])
    )
    recommendation = (
        "Prepare the critical alert(s) for risk analyst review."
        if critical else "Continue monitoring; consult desk policy if further review is needed."
    )
    return {"explanation": explanation, "recommendation": recommendation,
            "llm_status": llm_status}


def _approval_gate(state: InvestigationState):
    if state.get("instrument"):
        critical_rows = ([state["breach"]]
                         if state.get("breach", {}).get("status") == "Critical" else [])
    else:
        critical_rows = [
            row for row in state.get("breaches", []) if row.get("status") == "Critical"
        ]
    if state.get("breach", {}).get("status") == "Critical" and not critical_rows:
        critical_rows = [state["breach"]]
    if critical_rows:
        instrument = state.get("instrument") or (
            critical_rows[0]["instrument"] if len(critical_rows) == 1 else "Portfolio"
        )
        decision = interrupt({
            "type": "human_approval_required",
            "instrument": instrument,
            "summary": state.get("explanation"),
            "recommendation": state.get("recommendation"),
            "evidence": {**state.get("evidence", {}), "critical_positions": critical_rows},
        })
        return {"approval": decision, "status": "reviewed"}
    return {"approval": {"decision": "not_required", "comment": "No critical alert"},
            "status": "completed"}


def build_graph(enable_llm: bool = True):
    builder = StateGraph(InvestigationState)
    builder.add_node("plan", _plan)
    builder.add_node("execute", _execute)
    builder.add_node("explain", lambda state: _explain(state, enable_llm=enable_llm))
    builder.add_node("approval", _approval_gate)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda state: "execute" if state.get("selected_tools") else END,
        {"execute": "execute", END: END},
    )
    builder.add_edge("execute", "explain")
    builder.add_edge("explain", "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=MemorySaver())
