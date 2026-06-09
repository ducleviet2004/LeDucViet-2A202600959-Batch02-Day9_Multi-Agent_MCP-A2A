"""RAG Agent LangGraph definition.

Defines a custom graph that takes a search query via messages and calls the local RAG pipeline.
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    from legal_rag.src.task9_retrieval_pipeline import retrieve
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from legal_rag.src.task9_retrieval_pipeline import retrieve


class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def retrieve_docs_node(state: RAGState) -> dict:
    """Retrieve documents using the query from the last message."""
    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            query = msg.content
            break

    if not query:
        return {"messages": [AIMessage(content="[]")]}

    # Call RAG pipeline retrieve
    try:
        results = retrieve(query, top_k=5)
    except Exception as e:
        print(f"Error in RAG Agent retrieval: {e}")
        results = []

    # Serialize results to JSON
    response_text = json.dumps(results, ensure_ascii=False, indent=2)
    return {"messages": [AIMessage(content=response_text)]}


def create_graph():
    """Return a compiled LangGraph for RAG retrieval."""
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_docs_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile()
