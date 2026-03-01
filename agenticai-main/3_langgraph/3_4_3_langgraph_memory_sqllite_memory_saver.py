from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Step 1: Define stateful node ---
def manage_tasks(state):
    """Add or list tasks, persisted in SQLite memory"""
    action = state.get("action")
    tasks = state.get("tasks", [])
    
    if action == "add":
        task = state.get("task")
        tasks.append(task)
        state["tasks"] = tasks
        state["msg"] = f"Added: '{task}'. Total tasks: {len(tasks)}"
    
    elif action == "list":
        if tasks:
            task_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tasks))
            state["msg"] = f"My tasks:\n{task_list}"
        else:
            state["msg"] = "No tasks yet! Add one to get started."
    
    return state


# --- Step 2: Build graph ---
graph = StateGraph(dict)
graph.add_node("manage_tasks", manage_tasks)
graph.set_entry_point("manage_tasks")
graph.add_edge("manage_tasks", END)

# --- Step 3: Use SqliteSaver ---
session_id = "task_session_1"

with SqliteSaver.from_conn_string("c://code//agenticai//3_langgraph//checkpointer.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    
    def get_current_state():
        """Helper to retrieve the latest state from checkpoint"""
        checkpoint = checkpointer.get(config={"configurable": {"thread_id": session_id}})
        if checkpoint is None:
            return {}
        checkpoint_data = getattr(checkpoint, "checkpoint", checkpoint)
        channel_values = checkpoint_data.get("channel_values", {})
        return channel_values.get("__root__", {})

    # --- Run 1: Add first task ---
    print("\n--- Run 1: Add Task ---")
    state = get_current_state()
    state.update({"action": "add", "task": "Buy groceries"})
    result1 = app.invoke(state, config={"configurable": {"thread_id": session_id}})
    print(result1["msg"])

    # --- Run 2: Add another task ---
    print("\n--- Run 2: Add Another Task ---")
    state = get_current_state()
    state.update({"action": "add", "task": "Finish LangGraph tutorial"})
    result2 = app.invoke(state, config={"configurable": {"thread_id": session_id}})
    print(result2["msg"])

    # --- Run 3: List all tasks (memory restored!) ---
    print("\n--- Run 3: List Tasks ---")
    state = get_current_state()
    state.update({"action": "list"})
    result3 = app.invoke(state, config={"configurable": {"thread_id": session_id}})
    print(result3["msg"])