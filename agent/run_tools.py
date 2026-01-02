from agent.orchestrator import orchestrator
from agent.tools.search import web_search, fetch_arxiv
from agent.tools.rag import rag_search, rag_search_filter
from agent.tools.common import final_answer
from typing import cast
from langchain_core.agents import AgentAction
from agent.state import AgentState



# run_orchestrator(): main function that executes the orchestrator and processes its output to extract the relevant tool and its arguments.
# We'll use this information to update the state for future steps.
def run_orchestrator(state: dict) -> dict:
    '''Runs the orchestrator and processes the output to extract tool information.

    Args:
        state (dict): The current state containing the 'intermediate_steps'.

    Returns:
        dict: A new state with updated 'intermediate_steps' including the tool action.
    '''
    
    print('run_orchestrator')
    print(f'intermediate_steps: {state["intermediate_steps"]}')
    
    # Invoke the oraorchestratorcle with the current state.
    out = orchestrator.invoke(state)

    # Extract the tool name and its arguments from the orchestrator's response.
    tool_name = out.tool_calls[0]['name']
    tool_args = out.tool_calls[0]['args']

    # Create an AgentAction object, which records the tool used and the input provided.
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log='TBD'  # To be determined later after the tool runs.
    )

    # Return a new state with updated 'intermediate_steps'.
    return cast(AgentState,{
        'intermediate_steps': [action_out]
    })


# The router() function determines the next tool to use based on the current state.
def router(state: dict) -> str:
    '''Determines the next tool to use based on the current state.

    Args:
        state (dict): The current state containing 'intermediate_steps'.

    Returns:
        str: The name of the tool to use next.
    '''

    if isinstance(state['intermediate_steps'], list):
        return state['intermediate_steps'][-1].tool
    else:
        print('Router invalid format')
        return 'final_answer'


tool_str_to_func = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_arxiv': fetch_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}

# The run_tool() function executes the appropriate tool based on the current state.
def run_tool(state: dict) -> dict:
    '''Executes the appropriate tool based on the current state.

    Args:
        state (dict): The current state containing the 'intermediate_steps'.

    Returns:
        dict: A new state with updated 'intermediate_steps' including the tool's result.
    '''

    tool_name = state['intermediate_steps'][-1].tool
    tool_args = state['intermediate_steps'][-1].tool_input

    print(f'{tool_name}.invoke(input={tool_args})')

    out = tool_str_to_func[tool_name].invoke(input=tool_args)

    observation = str(out)

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )

    return {'intermediate_steps': [(action_out, observation)]}