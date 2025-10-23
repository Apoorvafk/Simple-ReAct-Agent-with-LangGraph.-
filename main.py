import os
import re
import json
import operator
from typing import TypedDict, Optional, Literal
from typing_extensions import Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


class AgentState(TypedDict, total=False):
    query: str
    needs_tool: Optional[bool]
    reason: Optional[str]
    tool_result: Optional[str]
    final_answer: Optional[str]
    # Persistent memory across turns (merged via dict union)
    facts: Annotated[dict, operator.or_]


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize the Gemini chat LLM via langchain-google-genai.

    Reads API key from `GOOGLE_API_KEY` and model name from `GEMINI_MODEL`.
    Defaults model to 'gemini-1.5-flash' if not provided.
    """
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=0.2)


def get_tavily() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Add it to .env and reload the environment."
        )
    return TavilyClient(api_key=key)


def parse_bool_json(s: str) -> tuple[Optional[bool], Optional[str]]:
    """Parse a minimal JSON blob for {"needs_tool": <bool>, "reason": <str>}.
    Returns (needs_tool, reason)."""
    try:
        data = json.loads(s)
        needs = data.get("needs_tool")
        reason = data.get("reason")
        if isinstance(needs, bool):
            return needs, reason if isinstance(reason, str) else None
    except Exception:
        pass
    return None, None


def reason_node(state: AgentState) -> AgentState:
    """
    Reasoning node:
    - If no tool_result is present: decide whether to use the tool or answer directly.
    - If tool_result is present: craft a final, readable answer using the tool result.
    """
    llm = get_llm()
    query = state["query"]

    # Lightweight memory update: extract user's name if provided in this query
    facts = dict(state.get("facts") or {})

    name_patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-\'\s]{0,40})\b",
        r"\bi am\s+([A-Za-z][A-Za-z\-\'\s]{0,40})\b",
        r"\bI'm\s+([A-Za-z][A-Za-z\-\'\s]{0,40})\b",
    ]
    for pat in name_patterns:
        m = re.search(pat, query, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().split()[0].strip(".,!?")
            if candidate and len(candidate) <= 40:
                facts["user_name"] = candidate
                break

    # If user asks for their name, answer from memory if available
    if re.search(r"\bwhat\s+is\s+my\s+name\b", query, flags=re.IGNORECASE):
        if facts.get("user_name"):
            return {"final_answer": f"Your name is {facts['user_name']}.", "needs_tool": False, "facts": facts}
        else:
            return {"final_answer": "I don't know your name yet. You can tell me by saying 'My name is ...'.", "needs_tool": False, "facts": facts}

    # If we have tool results, finalize the answer using them
    if state.get("tool_result"):
        prompt = (
            "You are a helpful assistant. The user asked a question. "
            "Use the provided tool result to produce a clear, concise, and readable answer.\n\n"
            f"User question: {query}\n\n"
            f"Tool result (verbatim, may be noisy):\n{state['tool_result']}\n\n"
            "Write a direct answer for the user. If multiple sources are present, synthesize them."
        )
        msg = llm.invoke(prompt)
        return {
            "final_answer": msg.content if hasattr(msg, "content") else str(msg),
            "needs_tool": False,
        }

    # Otherwise: decide whether we need to use the tool
    router_prompt = (
        "You are a routing assistant. Decide if answering the user's question requires a web search tool.\n"
        "Output only JSON with keys: needs_tool (true/false), reason (short).\n"
        "Use the tool when the question needs timely, factual, or unknown information that may require web results.\n\n"
        f"User question: {query}\n\n"
        '{"needs_tool": true/false, "reason": "..."}'
    )
    router_msg = llm.invoke(router_prompt)
    raw = router_msg.content if hasattr(router_msg, "content") else str(router_msg)
    needs_tool, reason = parse_bool_json(raw)

    # Fallback heuristic if LLM didn't return valid JSON
    if needs_tool is None:
        lower_q = query.lower()
        realtime_keywords = [
            "latest", "current", "today", "now", "live", "news", "trending",
            "price", "stock", "market cap", "exchange rate", "rate", "score", "result",
            "who won", "schedule", "forecast", "weather", "temperature", "traffic",
            "aqi", "air quality", "pollution", "covid", "cases", "update", "search", "browse",
        ]
        world_facts_keywords = ["population", "gdp", "area", "timezone", "capital of", "when is", "how many", "how much"]
        needs_tool = any(k in lower_q for k in realtime_keywords) or (
            any(k in lower_q for k in world_facts_keywords) and ("of" in lower_q or "in" in lower_q)
        )
        reason = reason or (
            "Heuristic: likely needs web search (real-time or factual lookup)."
            if needs_tool else "Heuristic: likely answerable without search."
        )

    # If no tool needed, generate the final answer directly
    if not needs_tool:
        answer_prompt = (
            "Answer the user's question clearly and concisely.\n\n"
            f"Question: {query}"
        )
        answer_msg = llm.invoke(answer_prompt)
        return {
            "needs_tool": False,
            "reason": reason,
            "final_answer": answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg),
            "facts": facts,
        }

    # Otherwise, ask the Execute node to run the tool
    return {"needs_tool": True, "reason": reason, "facts": facts}


def execute_node(state: AgentState) -> AgentState:
    """Execute the Tavily search tool and attach a compact textual result."""
    client = get_tavily()
    query = state["query"]
    # Perform a search; keep it compact to pass back to the Reason node
    result = client.search(query=query, max_results=5)

    # Create a succinct textual digest
    lines = []
    answer = result.get("answer") if isinstance(result, dict) else None
    if answer:
        lines.append(f"Tavily Answer: {answer}")

    results = result.get("results") if isinstance(result, dict) else None
    if isinstance(results, list):
        for i, r in enumerate(results[:3], start=1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or "(no url)"
            snippet = r.get("content") or r.get("snippet") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n{snippet}")

    digest = "\n\n".join(lines) if lines else json.dumps(result, ensure_ascii=False)
    return {"tool_result": digest}


def cleanup_node(state: AgentState) -> AgentState:
    """Clear ephemeral fields so each turn starts fresh, while preserving facts."""
    return {
        "tool_result": None,
        "needs_tool": None,
        "reason": None,
        "final_answer": None,
    }


def route_from_reason(state: AgentState) -> Literal["tool", "final"]:
    """
    Conditional edge router:
    - If needs_tool is True and we don't yet have a tool_result, go to Execute.
    - Otherwise, end.
    """
    if state.get("needs_tool") and not state.get("tool_result"):
        return "tool"
    return "final"


def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("cleanup", cleanup_node)
    graph.add_node("reason", reason_node)
    graph.add_node("execute", execute_node)
    
    graph.add_edge(START, "cleanup")
    graph.add_edge("cleanup", "reason")
    graph.add_conditional_edges(
        "reason",
        route_from_reason,
        {
            "tool": "execute",
            "final": END,
        },
    )
    # After executing the tool, return to reason to finalize
    graph.add_edge("execute", "reason")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app


def main():
    load_dotenv()
    # Validate keys up front for better DX
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set. Set it in .env for Gemini.")
    if not os.getenv("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not set. Set it in .env for Tavily.")

    app = build_graph()

    # Simple CLI loop
    print("Basic Reason/Act Agent (LangGraph)\nType 'exit' to quit.\n")
    while True:
        try:
            q = input("Your question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        # Use a static thread id for simplicity; in multi-user scenarios, make it unique per session
        thread = {"configurable": {"thread_id": "default"}}
        state: AgentState = {"query": q}
        final_state = app.invoke(state, thread)

        print("\n--- Answer ---")
        if final_state.get("final_answer"):
            print(final_state["final_answer"])
        else:
            # Fallback if something unexpected happened
            print("No final answer produced. State:")
            print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print("--------------\n")


if __name__ == "__main__":
    main()
