from typing import Annotated, Sequence, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the operational state of our Research Agent.
    - messages: Thread history with automatic message reducer (add_messages).
    - sources: Running log of citations and data sources accessed during the investigation.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

