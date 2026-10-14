"""LangGraph-style multi-agent router with a dependency-free fallback state machine."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: str
    answer: str
    history: list[str]


def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    if any(word in question for word in ("订单", "物流", "发货")):
        route = "order"
    elif any(word in question for word in ("退款", "退货", "换货")):
        route = "after_sales"
    else:
        route = "product"
    return {**state, "route": route, "history": state.get("history", []) + [f"route:{route}"]}


def run_workflow(question: str, product_agent, order_agent, after_sales_agent) -> AgentState:
    state = router_node({"question": question})
    agents = {"product": product_agent, "order": order_agent, "after_sales": after_sales_agent}
    agent = agents[state["route"]]
    state["answer"] = agent.handle(question)
    return state


def build_langgraph(product_agent, order_agent, after_sales_agent):
    from langgraph.graph import END, StateGraph
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("product", lambda state: {**state, "answer": product_agent.handle(state["question"])})
    graph.add_node("order", lambda state: {**state, "answer": order_agent.handle(state["question"])})
    graph.add_node("after_sales", lambda state: {**state, "answer": after_sales_agent.handle(state["question"])})
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda state: state["route"], {"product": "product", "order": "order", "after_sales": "after_sales"})
    for name in ("product", "order", "after_sales"):
        graph.add_edge(name, END)
    return graph.compile()
