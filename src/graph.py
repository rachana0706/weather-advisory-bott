from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.nodes import (
    extract_intent,
    fetch_weather_node,
    ask_for_location,
    api_failure_response,
    match_sop_node,
    generate_fact_based_advice,
    no_guidance_response
)

def has_location(state: AgentState) -> str:
    if state.get("current_location"):
        return "fetch_weather"
    return "ask_for_location"

def weather_success(state: AgentState) -> str:
    if state.get("weather_error"):
        return "api_failure_response"
    return "match_sop"

def sops_found(state: AgentState) -> str:
    if state.get("selected_sop"):
        return "generate_advice"
    return "no_guidance"

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("extract_intent", extract_intent)
    workflow.add_node("ask_for_location", ask_for_location)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("api_failure_response", api_failure_response)
    workflow.add_node("match_sop", match_sop_node)
    workflow.add_node("generate_advice", generate_fact_based_advice)
    workflow.add_node("no_guidance", no_guidance_response)
    
    workflow.set_entry_point("extract_intent")
    
    workflow.add_conditional_edges(
        "extract_intent",
        has_location,
        {
            "fetch_weather": "fetch_weather",
            "ask_for_location": "ask_for_location"
        }
    )
    
    workflow.add_edge("ask_for_location", END)
    
    workflow.add_conditional_edges(
        "fetch_weather",
        weather_success,
        {
            "api_failure_response": "api_failure_response",
            "match_sop": "match_sop"
        }
    )
    
    workflow.add_edge("api_failure_response", END)
    
    workflow.add_conditional_edges(
        "match_sop",
        sops_found,
        {
            "generate_advice": "generate_advice",
            "no_guidance": "no_guidance"
        }
    )
    
    workflow.add_edge("generate_advice", END)
    workflow.add_edge("no_guidance", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

graph = create_graph()
