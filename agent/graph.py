from langgraph.graph import StateGraph, END
from agent.orchestrator import run_orchestrator
from agent.run_tools import run_tool, router
from agent.state import AgentState
from agent.tools.search import web_search, fetch_arxiv
from agent.tools.rag import rag_search, rag_search_filter
from agent.tools.common import final_answer



possible_next_nodes = [
    'rag_search_filter',
    'rag_search',
    'fetch_arxiv',
    'web_search',
    'final_answer',
]

tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer

]

# Initialize the state graph with AgentState to manage the workflow.
graph = StateGraph(AgentState)

graph.add_node('orchestrator', run_orchestrator)
graph.add_node('rag_search_filter', run_tool)
graph.add_node('rag_search', run_tool)
graph.add_node('fetch_arxiv', run_tool)
graph.add_node('web_search', run_tool)
graph.add_node('final_answer', run_tool)

# Set the entry point to 'orchestrator'.
graph.set_entry_point('orchestrator')

# # Add conditional edges to determine the next step using the router function.
# graph.add_conditional_edges(source='orchestrator', 
#                             path=router,
#                             nodes=possible_next_nodes 
#                             

graph.add_conditional_edges(
    'orchestrator',              # Positional: 1st argument (source)
    router,                      # Positional: 2nd argument (path function)
    possible_next_nodes          # Positional: 3rd argument (destinations)
)

# Add edges from each tool back to 'orchestrator', except 'final_answer', which leads to 'END'.
for tool_obj in tools:
    if tool_obj.name != 'final_answer':
        graph.add_edge(tool_obj.name, 'orchestrator')

graph.add_edge('final_answer', END)

# Compile the graph to make it executable.
runnable = graph.compile()
