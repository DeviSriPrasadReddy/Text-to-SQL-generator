# graph/builder.py
# -----------------
# Assembles the graph by adding nodes, edges, and compiling it.

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from graph.state import AgentState
from graph.nodes import enhance_question, call_model, should_continue
from tools.sql_tools import all_tools

def create_graph():
    """
    Creates and compiles the stateful Text-to-SQL graph.
    """
    print("--- Compiling Graph ---")
    
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # 1. Add the nodes
    workflow.add_node("enhance_question", enhance_question) # New entry point
    workflow.add_node("agent", call_model)                 # The "brain"
    workflow.add_node("call_tools", ToolNode(all_tools))   # The "hands"

    # 2. Define the edges
    
    # Start at the enhancer
    workflow.set_entry_point("enhance_question")
    
    # Enhancer always goes to the agent
    workflow.add_edge("enhance_question", "agent")

    # The agent decides what to do next
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "call_tools": "call_tools",
            "END": END
        }
    )
    
    # The tools node always goes back to the agent to process results
    workflow.add_edge("call_tools", "agent")

    # 3. Compile the graph with memory
    
    # We use in-memory persistence for this example.
    # For production, you'd use a persistent DB.
    memory = SqliteSaver.from_conn_string(":memory:")
    
    app = workflow.compile(checkpointer=memory)
    
    print("--- Graph Compiled ---")
    return app
