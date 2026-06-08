from langgraph.graph import StateGraph, START, END
from src.state.match_state import MatchAnalysisState
from src.agents.reporting.reporting_agent import ReportingAgent

def video_processor_node(state: MatchAnalysisState):
    print("[VideoProcessor] Assuming events are already processed and available in events.json")
    return {"events_file": state.get("events_file", "output/events.json")}

def reporting_node(state: MatchAnalysisState):
    agent = ReportingAgent()
    # It reads events_file from state
    report = agent.generate_report(state["events_file"])
    return {"report": report}

def build_match_graph():
    """
    Build and compile a simplified match analysis graph.
    START -> video_processor -> reporting -> END
    """
    graph = StateGraph(MatchAnalysisState)

    graph.add_node("video_processor", video_processor_node)
    graph.add_node("reporting", reporting_node)

    graph.add_edge(START, "video_processor")
    graph.add_edge("video_processor", "reporting")
    graph.add_edge("reporting", END)

    compiled = graph.compile()

    print("[MatchGraph] Compiled: START -> video_processor -> reporting -> END")

    return compiled

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    root_path = Path(__file__).parent.parent.parent.absolute()
    events_path = root_path / "output" / "events.json"
    
    if not events_path.exists():
        print(f"Error: {events_path} no existe.")
    else:
        graph = build_match_graph()
        initial_state = {"events_file": str(events_path)}
        result = graph.invoke(initial_state)
        print("\n=== REPORTE GENERADO ===")
        print(result["report"])
