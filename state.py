# graph/state.py
# -----------------
# Defines the state object that is passed between nodes in the graph.

from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The state of our agent.
    
    Attributes:
        messages: The conversation history.
        schema_info: The database schema provided as metadata.
    """
    messages: Annotated[list, lambda x, y: x + y]
    schema_info: str
