# graph/nodes.py
# -----------------
# (Imports are the same...)

from databricks_langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from graph.state import AgentState
from tools.sql_tools import all_tools
import config
from utils.callbacks import TokenUsageCallback # We keep our callback

# --- 1. Initialize LLMs and Bind Tools ---
enhancer_llm = ChatDatabricks(...) # (unchanged)
sql_agent_llm = ChatDatabricks(...) # (unchanged)

# --- (MODIFIED) ---
# We bind the *new* (smaller) list of tools
llm_with_tools = sql_agent_llm.bind_tools(all_tools)

token_callback = TokenUsageCallback() # (unchanged)


# --- 2. Define Node Functions ---

def enhance_question(state: AgentState):
    """
    Node: Rewrites the user's question.
    (This function is completely unchanged.)
    """
    print("--- Calling Node: enhance_question ---")
    # ... (no changes here)
    messages = state["messages"]
    prompt = f"""
    You are a query rewriting assistant...
    ...
    **Enhanced Question:**
    """
    response = enhancer_llm.invoke(
        prompt, 
        config={"callbacks": [token_callback]}
    )
    enhanced_q = response.content.strip()
    print(f"Original question: {messages[-1].content}")
    print(f"Enhanced question: {enhanced_q}")
    enhancer_message = HumanMessage(content=enhanced_q)
    return {"messages": [enhancer_message]}


# --- (MAJOR MODIFICATION) ---
def call_model(state: AgentState) -> dict:
    """
    Node: The main "brain" of the agent.
    It now receives the schema directly from the state.
    """
    print("--- Calling Node: call_model (SQL Agent) ---")
    messages = state["messages"]
    
    # Get the schema from the state
    schema_info = state["schema_info"] 
    
    system_prompt = f"""
    You are an expert Databricks SQL assistant.
    Your goal is to answer the user's question by generating and executing SQL
    against the '{config.CATALOG}.{config.SCHEMA}' database.

    **The database schema is provided to you below:**
    --- SCHEMA START ---
    {schema_info}
    --- SCHEMA END ---

    **Your task:**
    1.  Read the LAST message in the conversation (the 'Enhanced Instruction').
    2.  Use the schema above to write a single, correct SQL query to answer 
        that instruction.
    3.  Call the `execute_sql_query` tool with your generated query.
    
    **After the query is executed:**
    - If it's successful, use the results to provide a final, 
      natural-language answer.
    - If it fails, analyze the error, correct your SQL, and call 
      `execute_sql_query` again.
    """
    
    messages_with_prompt = [SystemMessage(content=system_prompt)] + messages
    
    response = llm_with_tools.invoke(
        messages_with_prompt, 
        config={"callbacks": [token_callback]}
    )
    
    return {"messages": [response]}


# --- (UNCHANGED) ---
def should_continue(state: AgentState) -> str:
    """
    This function is unchanged. It still routes to 'call_tools'
    if 'execute_sql_query' was called, or to 'END' for a final answer.
    """
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        print("--- Decision: Call Tools ---")
        return "call_tools"
    
    print("--- Decision: End ---")
    return "END"
