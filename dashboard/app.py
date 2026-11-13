import sys
from uuid import uuid4
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
from risk_engine.io import load_positions
from risk_engine.exposure import portfolio_exposure
from risk_engine.concentration import concentration_analysis
from risk_engine.limit_detection import detect_limit_breaches
from risk_engine.var_calculator import historical_var, generate_synthetic_returns
from risk_engine.stress_testing import stress_test
from agent.graph import build_graph
from database.audit import record_decision, get_decisions
from langgraph.types import Command

st.set_page_config(page_title="TradeGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ TradeGuard")
st.caption("Prototype trading-risk monitoring system · Synthetic data only · Not for production decisions")

df = load_positions()
exposure = portfolio_exposure(df)
breaches = detect_limit_breaches(df)
var = historical_var(df)
concentration = concentration_analysis(df)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Portfolio exposure", f"₹{exposure['total_exposure']/100000:.2f} L")
c2.metric("95% one-day VaR", f"₹{var['var']/100000:.2f} L")
c3.metric("Critical positions", sum(x["status"]=="Critical" for x in breaches))
c4.metric("Warning positions", sum(x["status"]=="Warning" for x in breaches))

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Portfolio","Risk alerts","Investigate","Approval","Audit log"])
with tab1:
    st.subheader("Trading positions")
    st.dataframe(df, use_container_width=True)
    st.subheader("Sector exposure")
    st.dataframe(pd.DataFrame(exposure["by_sector"]), use_container_width=True)
    st.subheader("Concentration flags (>40%)")
    st.json(concentration)
    st.subheader("Stress test")
    shock = st.slider("Parallel market shock (%)", -30, 10, -5)
    result = stress_test(df, market_shock=shock/100)
    st.metric("Estimated P&L under scenario", f"₹{result['estimated_portfolio_pnl']:,.0f}")
with tab2:
    st.subheader("Deterministic limit monitoring")
    alerts = pd.DataFrame(breaches)
    st.dataframe(alerts, use_container_width=True)
    st.caption("Normal <90% utilization · Warning 90–100% · Critical >100% (sample policy)")
with tab3:
    st.subheader("Agent investigation")
    question = st.text_input("Ask a risk question", "Why has Infosys triggered a trading risk alert?")
    approval_pending = bool(st.session_state.get("pending"))
    if st.button("Run investigation", disabled=approval_pending):
        graph = build_graph()
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = graph.invoke({"question": question}, config=config)
        st.session_state["graph"] = graph
        st.session_state["config"] = config
        st.session_state["investigation"] = result
        st.session_state["pending"] = result.get("__interrupt__", [])
        st.rerun()
    result = st.session_state.get("investigation")
    if result:
        st.write(result.get("explanation", ""))
        st.write("Recommendation:", result.get("recommendation", ""))
        st.json({"selected_tools": result.get("selected_tools"),
                 "tool_calls": result.get("tool_calls"),
                 "evidence": result.get("evidence"),
                 "status": result.get("status")})
        if result.get("__interrupt__"):
            st.warning("A human approval is required in the Approval tab.")
with tab4:
    st.subheader("Human approval checkpoint")
    pending = st.session_state.get("pending", [])
    if pending:
        payload = pending[0].value
        st.json(payload)
        reviewer = st.text_input("Reviewer", "Risk Analyst")
        comment = st.text_area("Review comment")
        col_a,col_b = st.columns(2)
        decision = None
        if col_a.button("Approve escalation"):
            decision = "approved"
        if col_b.button("Reject escalation"):
            decision = "rejected"
        if decision:
            graph = st.session_state["graph"]
            config = st.session_state["config"]
            resumed = graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer, "comment": comment}), config=config)
            record_decision(payload.get("instrument") or "Portfolio", decision,
                            reviewer, comment, payload.get("evidence", {}))
            st.session_state["investigation"] = resumed
            st.session_state["pending"] = []
            st.success(f"Decision recorded: {decision}")
    else:
        st.info("No pending approval. Run an investigation for a critical position.")
with tab5:
    st.subheader("Recorded human decisions")
    st.dataframe(pd.DataFrame(get_decisions()), use_container_width=True)
