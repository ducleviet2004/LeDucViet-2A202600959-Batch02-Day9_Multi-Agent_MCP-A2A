"""Stage 4: Multi-Agent System (Optimized & Instrumented)

Contains structural and logic optimizations to reduce latency.
Graph:
    START -> analyze_law -> aggregate
    START -> check_routing -> parallel [call_tax_specialist, call_compliance_specialist] -> aggregate -> END
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
from langchain_core.tools import tool
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
# Node implementations (Optimized)
# ---------------------------------------------------------------------------

async def analyze_law(state: LegalState) -> dict:
    """Lead attorney analyses the legal aspects of the question."""
    t0 = time.time()
    print("\n  [Node OPT: analyze_law] Lead attorney analysing legal aspects...")
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
    print(f"  [Node OPT: analyze_law] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "law_analysis": result.content,
        "node_timings": {"analyze_law": elapsed}
    }

async def check_routing(state: LegalState) -> dict:
    """Routing node: determine which specialist sub-agents are needed using fast-path keyword checks."""
    t0 = time.time()
    print("\n  [Node OPT: check_routing] Determining which specialists are needed...")
    
    question_lower = state["question"].lower()
    needs_tax = False
    needs_compliance = False
    
    # 1. Fast Keyword Routing Optimization
    if any(kw in question_lower for kw in ["tax", "irs", "evasion", "penalt", "fbar", "fatca", "thuế"]):
        needs_tax = True
    if any(kw in question_lower for kw in ["compliance", "sec", "sox", "aml", "fcpa", "regulatory", "governance", "tuân thủ"]):
        needs_compliance = True
        
    if needs_tax or needs_compliance:
        elapsed = time.time() - t0
        print(f"  [Node OPT: check_routing] (Fast-Path Keyword Match) needs_tax={needs_tax}, needs_compliance={needs_compliance} in {elapsed*1000:.2f}ms")
        return {
            "needs_tax": needs_tax,
            "needs_compliance": needs_compliance,
            "node_timings": {"check_routing": elapsed}
        }

    # Fallback to LLM routing if keywords do not match
    print("  [Node OPT: check_routing] No keywords matched, falling back to LLM...")
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
    print(f"  [Node OPT: check_routing] (LLM Path) needs_tax={needs_tax}, needs_compliance={needs_compliance} in {elapsed:.2f}s")
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
    """Tax specialist sub-agent (Optimized: Direct Prompt Retrieval instead of ReAct loop)."""
    t0 = time.time()
    print("\n  [Node OPT: call_tax_specialist] Tax specialist starting (direct mode)...")

    # Run tool directly in python
    retrieved_laws = search_tax_law(state["question"])
    
    tax_prompt = (
        "You are a specialist tax attorney and CPA with expertise in corporate tax law, "
        "tax evasion vs. avoidance, IRS enforcement, penalties under IRC §§ 6651/6662/6663, "
        "FBAR/FATCA requirements, and tax fraud statutes (18 U.S.C. § 7201-7207).\n\n"
        "Here are the relevant sections from the Tax Law database:\n"
        f"{retrieved_laws}\n\n"
        "Analyse the tax aspects of the question based on the retrieved laws above. Keep your response under 200 words."
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=tax_prompt),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(f"  [Node OPT: call_tax_specialist] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "tax_result": result.content,
        "node_timings": {"call_tax_specialist": elapsed}
    }

async def call_compliance_specialist(state: LegalState) -> dict:
    """Compliance specialist sub-agent (Optimized: Direct Prompt Retrieval instead of ReAct loop)."""
    t0 = time.time()
    print("\n  [Node OPT: call_compliance_specialist] Compliance specialist starting (direct mode)...")

    # Run tool directly in python
    retrieved_laws = search_compliance_law(state["question"])
    
    compliance_prompt = (
        "You are a senior regulatory compliance officer with expertise in SEC enforcement, "
        "SOX compliance, FTC regulations, FCPA, AML/BSA, GDPR, CCPA, and corporate governance.\n\n"
        "Here are the relevant sections from the Regulatory Compliance Law database:\n"
        f"{retrieved_laws}\n\n"
        "Analyse the compliance aspects of the question based on the retrieved laws above. Keep your response under 200 words."
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=compliance_prompt),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(f"  [Node OPT: call_compliance_specialist] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "compliance_result": result.content,
        "node_timings": {"call_compliance_specialist": elapsed}
    }

async def aggregate(state: LegalState) -> dict:
    """Combine all specialist analyses into a final comprehensive answer."""
    t0 = time.time()
    print("\n  [Node OPT: aggregate] Combining all specialist analyses...")
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
    print(f"  [Node OPT: aggregate] Done in {elapsed:.2f}s ({len(result.content)} chars)")
    return {
        "final_answer": result.content,
        "node_timings": {"aggregate": elapsed}
    }

# ---------------------------------------------------------------------------
# Graph construction (Optimized Parallel Topology)
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the optimized multi-agent StateGraph."""
    graph = StateGraph(LegalState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax_specialist", call_tax_specialist)
    graph.add_node("call_compliance_specialist", call_compliance_specialist)
    graph.add_node("aggregate", aggregate)

    # Parallel starting nodes
    graph.add_edge(START, "analyze_law")
    graph.add_edge(START, "check_routing")
    
    # Analyze law proceeds straight to aggregate
    graph.add_edge("analyze_law", "aggregate")
    
    # check_routing dynamically routes to specialists or aggregate
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "aggregate"],
    )
    
    # Specialists proceed to aggregate
    graph.add_edge("call_tax_specialist", "aggregate")
    graph.add_edge("call_compliance_specialist", "aggregate")
    
    # Aggregate ends the graph
    graph.add_edge("aggregate", END)

    return graph.compile()

QUESTION = "If a company breaks a contract and avoids taxes, what are the legal and regulatory consequences?"

async def main():
    print("=" * 70)
    print("STAGE 4: Multi-Agent System (OPTIMIZED & INSTRUMENTED)")
    print("=" * 70)
    print(f"Question: {QUESTION}\n")

    graph = create_graph()
    
    start_total = time.time()
    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "tax_result": "",
        "compliance_result": "",
        "final_answer": "",
        "node_timings": {},
    })
    total_time = time.time() - start_total

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["final_answer"])
    
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN (OPTIMIZED)")
    print("=" * 70)
    for node, duration in result.get("node_timings", {}).items():
        print(f"  - Node {node}: {duration:.2f}s")
    print(f"Total E2E Latency: {total_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
