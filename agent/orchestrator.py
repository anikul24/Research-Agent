from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./cred.env")

# Import the tools to bind them to the LLM
from langchain_core.tools import tool
from agent.tools.search import web_search, fetch_arxiv
from agent.tools.rag import rag_search, rag_search_filter
from agent.tools.common import final_answer

# Define the system prompt 
system_prompt = (
    '''You are the Agent LLM, the great AI decision-maker.
    Given the user's query, you must decide what to do with it based on the
    list of tools provided to you.

    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice (i.e., if the tool appears in the scratchpad twice, do
    not use it again).

    You should aim to collect information from a diverse range of sources before
    providing the answer to the user. Once you have collected plenty of information
    to answer the user's question (stored in the scratchpad), use the final_answer tool.'''
)


messages = [('system', system_prompt),

    ## Insert past chat messages to maintain context
    MessagesPlaceholder(variable_name='chat_history'),

    ('human', '{input}'),

    #scratchpad to track tool usage and intermediate steps
    ('assistant', 'scratchpad: {scratchpad}'),
    ]

prompt = ChatPromptTemplate.from_messages(messages)

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0,  openai_api_key = os.getenv('OPENAI_API_KEY'))

tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer

]

def create_scratchpad(intermediate_steps: list[ToolCall]) -> str:
    '''Create a scratchpad from the intermediate tool calls'''
    print('NEW create_scratchpad')
    print(f'intermediate_steps', intermediate_steps)

    research_steps = []

    # Iterate over the objects as a single variable
    for action_obj in intermediate_steps:
        # Check if the log property is not 'TBD' (meaning the action has been executed)
        log_str = action_obj.log
        if log_str != 'TBD':
            research_steps.append(
                f'Tool: {action_obj.tool}, input: {action_obj.tool_input}\n'
                f'Output: {log_str}'
            )
    
    # Join the research steps into a readable log.
    return '\n---\n'.join(research_steps)


#define orchestrator for decision making pipeline

orchestrator = (
    {
        'input': lambda x: x['input'],
        'chat_history': lambda x: x['chat_history'],
        'scratchpad': lambda x: create_scratchpad(intermediate_steps=x['intermediate_steps']),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice='any')
)