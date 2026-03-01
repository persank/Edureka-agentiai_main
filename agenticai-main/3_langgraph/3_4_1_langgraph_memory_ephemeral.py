from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    x: int
    path: str

def start(state: MyState) -> MyState:
    print("Starting workflow...")
    state["x"] = 10
    return state

def double(state: MyState) -> MyState:
    print("Doubling x...")
    state["x"] = state["x"] * 2
    state["path"] = "doubled"
    return state

def square(state: MyState) -> MyState:
    print("Squaring x...")
    state["x"] = state["x"] ** 2
    state["path"] = "squared"
    return state

def finalize(state: MyState) -> MyState:
    print(f"Final value: {state['x']} (via {state['path']} path)")
    return state

# Note: route_logic is not a node
# It is a routing function
def route_logic(state: MyState) -> str:
    # Route based on whether x is even or odd
    return "double" if state["x"] % 2 == 0 else "square"

# Build graph with conditional routing
workflow = StateGraph(MyState)

workflow.add_node("start", start)
workflow.add_node("double", double)
workflow.add_node("square", square)
workflow.add_node("finalize", finalize)

workflow.set_entry_point("start")
workflow.add_conditional_edges("start", route_logic, {"double": "double", "square": "square"})
workflow.add_edge("double", "finalize")
workflow.add_edge("square", "finalize")
workflow.add_edge("finalize", END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n=== Run 1: x=10 (even) ===")
    result1 = app.invoke({})
    print(f"Result: {result1}\n")
    
    print("=== Run 2: x=7 (odd) ===")
    # Modify start function to return 7
    def start_odd(state: MyState) -> MyState:
        print("Starting workflow...")
        state["x"] = 7
        return state
    
    workflow2 = StateGraph(MyState)
    workflow2.add_node("start", start_odd)
    workflow2.add_node("double", double)
    workflow2.add_node("square", square)
    workflow2.add_node("finalize", finalize)
    workflow2.set_entry_point("start")
    workflow2.add_conditional_edges("start", route_logic, {"double": "double", "square": "square"})
    workflow2.add_edge("double", "finalize")
    workflow2.add_edge("square", "finalize")
    workflow2.add_edge("finalize", END)
    
    app2 = workflow2.compile()
    result2 = app2.invoke({})
    print(f"Result: {result2}")