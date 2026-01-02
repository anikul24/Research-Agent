from langchain.tools import tool
from pydantic.v1 import BaseModel, Field
from semantic_router.encoders import OpenAIEncoder 
from openai import OpenAI
from dotenv import load_dotenv
import os
from agent.ingestion import get_pinecone_index

import time

load_dotenv(dotenv_path="./cred.env")


index_name = 'langgraph-research-agent'



client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

encoder = OpenAIEncoder(name='text-embedding-3-small')



index = get_pinecone_index(index_name=index_name)



def format_rag_text(matches: list) -> str:
    '''Formats the input text for the RAG tool and returns the formatted text as a string.'''
    formatted_text = []


    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Chunk: {x['metadata']['chunk']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
        )

        formatted_text.append(text)

    formatted_text = '\n---\n'.join(formatted_text)

    return formatted_text


class WebSearchArgs(BaseModel):
    arvix_id: str = Field(..., description="The arvix id string.")

@tool('rag_search_filter')
def rag_search_filter(query:str, arvix_id:str) -> str:
    '''RAG serach filter based on arvix id'''

    query_encode = encoder([query])

    input_vector = index.query(vector=query_encode, top_k=5, include_metadata=True, filter={'arxiv_id': arvix_id})

    matches = input_vector['matches']

    formatted_text = format_rag_text(matches)

    return formatted_text



@tool('rag_search')
def rag_search(query:str) -> str:
    '''RAG search without filter'''

    query_encode = encoder([query])

    input_vector = index.query(vector=query_encode, top_k=5, include_metadata=True)

    matches = input_vector['matches']

    formatted_text = format_rag_text(matches)

    return formatted_text