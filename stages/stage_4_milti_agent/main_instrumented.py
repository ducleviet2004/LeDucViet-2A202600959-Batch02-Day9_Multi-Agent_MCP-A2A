"""Stage 4: Multi-Agent System (Instrumented Unoptimized)

Identical to the original main.py, but instrumented with timing code
to measure step-by-step latency for accurate comparisons.
"""

import asyncio
import json
import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm
from stages.stage_4_milti_agent.main import search_tax_law, search_compliance_law

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a

def _merge_dict(a: dict, b: dict) -> dict:
    """Reducer: merge dictionary keys."""
    res = dict(a) if a else {}
    if b:
        res.update(b)
    return res

class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str
    node_timings: Annotated[dict, _merge_dict]

# ---------------------------------------------------------------------------
# Node implementations (Unoptimized, but with Timing)
# ---------------------------------------------------------------------------

async def analyze_law(state: LegalState) -> dict:
    """Lead attorney analyses the legal aspects of the question."""
    t0 = time.time()
    print("\n  [Node UNOPT: analyze_law] Lead attorney analysing legal aspects...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly. Keep your analysis under 200 words."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(f"  [Node UNOPT: analyze_law] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "law_analysis": result.content,
        "node_timings": {"analyze_law": elapsed}
    }

async def check_routing(state: LegalState) -> dict:
    """Routing node: determine which specialist sub-agents are needed using LLM (no keyword check)."""
    t0 = time.time()
    print("\n  [Node UNOPT: check_routing] Determining which specialists are needed...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'You are a legal routing expert. Based on the question, decide whether '
                'specialist sub-agents are needed.\n'
                'Reply with ONLY valid JSON — no markdown, no extra text:\n'
                '{"needs_tax": <true|false>, "needs_compliance": <true|false>}\n\n'
                'needs_tax = true  → question involves tax law, IRS, tax evasion, penalties\n'
                'needs_compliance = true → question involves regulatory compliance, SEC, SOX, AML, FCPA'
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"needs_tax": True, "needs_compliance": True}

    needs_tax = bool(parsed.get("needs_tax", True))
    needs_compliance = bool(parsed.get("needs_compliance", True))
    elapsed = time.time() - t0
    print(f"  [Node UNOPT: check_routing] needs_tax={needs_tax}, needs_compliance={needs_compliance} in {elapsed:.2f}s")
    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "node_timings": {"check_routing": elapsed}
    }

def route_to_specialists(state: LegalState) -> list[Send]:
    """Routing function: dispatch parallel Send objects to specialist nodes."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends

async def call_tax_specialist(state: LegalState) -> dict:
    """Tax specialist sub-agent (runs as inline ReAct agent)."""
    t0 = time.time()
    from langgraph.prebuilt import create_react_agent
    print("\n  [Node UNOPT: call_tax_specialist] Tax specialist agent starting...")

    tax_prompt = (
        "You are a specialist tax attorney and CPA with expertise in corporate tax law, "
        "tax evasion vs. avoidance, IRS enforcement, penalties under IRC §§ 6651/6662/6663, "
        "FBAR/FATCA requirements, and tax fraud statutes (18 U.S.C. § 7201-7207). "
        "Use the search_tax_law tool to ground your analysis. Keep your response under 200 words."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_tax_law], prompt=tax_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    elapsed = time.time() - t0
    print(f"  [Node UNOPT: call_tax_specialist] Done in {elapsed:.2f}s ({len(final_msg)} chars)")
    return {
        "tax_result": final_msg,
        "node_timings": {"call_tax_specialist": elapsed}
    }

async def call_compliance_specialist(state: LegalState) -> dict:
    """Compliance specialist sub-agent (runs as inline ReAct agent)."""
    t0 = time.time()
    from langgraph.prebuilt import create_react_agent
    print("\n  [Node UNOPT: call_compliance_specialist] Compliance specialist agent starting...")

    compliance_prompt = (
        "You are a senior regulatory compliance officer with expertise in SEC enforcement, "
        "SOX compliance, FTC regulations, FCPA, AML/BSA, GDPR, CCPA, and corporate governance. "
        "Use the search_compliance_law tool to ground your analysis. Keep your response under 200 words."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_compliance_law], prompt=compliance_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    elapsed = time.time() - t0
    print(f"  [Node UNOPT: call_compliance_specialist] Done in {elapsed:.2f}s ({len(final_msg)} chars)")
    return {
        "compliance_result": final_msg,
        "node_timings": {"call_compliance_specialist": elapsed}
    }

async def aggregate(state: LegalState) -> dict:
    """Combine all specialist analyses into a final comprehensive answer."""
    t0 = time.time()
    print("\n  [Node UNOPT: aggregate] Combining all specialist analyses...")
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response. Combine the following analyses "
                "into a cohesive answer with clear sections. Avoid redundancy. "
                "Keep your response under 500 words."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(f"  [Node UNOPT: aggregate] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "final_answer": result.content,
        "node_timings": {"aggregate": elapsed}
    }

# ---------------------------------------------------------------------------
# Graph construction (Original Sequential Topology)
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the original multi-agent StateGraph."""
    graph = StateGraph(LegalState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax_specialist", call_tax_specialist)
    graph.add_node("call_compliance_specialist", call_compliance_specialist)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "aggregate"],
    )
    graph.add_edge("call_tax_specialist", "aggregate")
    graph.add_edge("call_compliance_specialist", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
