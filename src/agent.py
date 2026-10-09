from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import AgentState
from .tools import search_private_docs, search_web

# 1. System Prompt instructing the Agent on tool selection & citation
SYSTEM_PROMPT = """You are 'LocalLens', an expert AI Research Assistant.
You have access to two tools:
1. `search_private_docs`: Use this FIRST whenever the user asks about confidential projects, internal documents, company specs, or uploaded files.
2. `search_web`: Use this when the query asks about real-time events, public news, documentation, or facts outside internal documents.

Rules:
- Be analytical, factual, and clear.
- Always cite your source (e.g., "[Source: project_apollo_specs.txt]" or "[Web: URL]").
- If private documents contain the answer, prioritize them over web information.
- If no tool is needed for general pleasantries or simple formatting, answer directly.
"""

# 2. Setup Ollama LLM and Bind Tools
tools = [search_private_docs, search_web]
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# 3. Define Graph Nodes
def agent_node(state: AgentState):
    """Executes the LLM with the accumulated message history."""
    messages = state["messages"]
    
    # Prepend system instruction if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 4. Construct LangGraph Workflow
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges & Conditional Routing
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)  # If LLM generated tool calls -> route to 'tools', else -> END
workflow.add_edge("tools", "agent")                      # Return tool outputs back to LLM to formulate final response

# Compile the Executable Agent
research_agent = workflow.compile()


def query_agent(prompt: str) -> str:
    """Convenience function to run a query and print step-by-step reasoning."""
    print(f"\n User Query: {prompt}\n" + "-"*50)
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    final_output = ""
    for event in research_agent.stream(initial_state, stream_mode="values"):
        last_msg = event["messages"][-1]
        
        # Display Tool Calls if LLM decided to invoke any
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f" [Agent Decision] Calling Tool: `{tc['name']}` with query: \"{tc['args'].get('query', '')}\"")
        
        # Capture final answer
        if last_msg.type == "ai" and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            final_output = last_msg.content
            
    print("\n Final Answer:\n", final_output)
    return final_output


if __name__ == "__main__":
    # Test sample query
    query_agent("What is the battery flight time and budget for Project Apollo?")

