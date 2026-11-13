# Live Groq Integration Test

**Result: PASS**  
**Date:** 2026-11-13

- **Provider/model:** Groq / `openai/gpt-oss-20b`; `.env` supplied the API key (presence verified; value never printed).
- **LLM:** Successful non-empty response from the configured model; deterministic fallback was not used.
- **LangGraph tools called:** `lookup_position`, `check_limit_breaches`, `retrieve_policy`.
- **Deterministic Infosys results:** market value `1,800,000`; configured limit `1,500,000`; excess `300,000`; utilization `1.2` (120%); classification `Critical`.
- **RAG:** ChromaDB returned distinct policy sections: Position limits, Escalation and human review, Evidence and auditability. Source: `data/policies/risk_policy.txt`.
- **Explanation checks:** non-empty; preserved the exact calculated values; no unverifiable percentage/threshold claims found.
- **Approval workflow:** graph paused for approval; pending state disabled a new investigation; resumed with approval.
- **Audit:** one approved Infosys record verified in an isolated temporary SQLite database.
- **Dashboard:** Streamlit AppTest ran all five tabs (Portfolio, Risk alerts, Investigate, Approval, Audit log) with no exceptions.
- **Additional checks:** `python -m compileall -q .` passed; project virtual environment pytest: `20 passed`.

The isolated SQLite database was used to avoid changing the project's audit history. No API key was included in the report.
