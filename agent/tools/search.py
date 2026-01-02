from dotenv import load_dotenv
import pydantic
from serpapi import GoogleSearch
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field # Used for tool schema
import os
import requests
import re

load_dotenv(dotenv_path="./cred.env")

SERP_API_KEY = os.getenv('SERP_API_KEY')

serp_params = {
  "engine": "google",
  "api_key": SERP_API_KEY
}




class ArxivInput(BaseModel):
    """Input for fetching an arXiv abstract."""
    # Renamed the class for clarity, matching the tool's purpose
    arvix_id: str = Field(description="The arXiv ID (e.g., '2407.03964') for the paper.")



# Regex pattern to find the abstract block
# Note: This pattern is highly specific to a newer arXiv layout.
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)


# Define the 'web_search' tool using the '@tool' decorator.
@tool('web_search')
def web_search(query: str) -> str:
    '''Finds general knowledge information using a Google search.

    Args:
        query (str): The search query string.
    
    Returns:
        str: A formatted string of the top search results, including title, snippet, and link.
    '''

    search = GoogleSearch({
        **serp_params,  
        'q': query,        
        'num': 5         
    })
   
    results = search.get_dict().get('organic_results', [])
    formatted_results = '\n---\n'.join(
        ['\n'.join([x['title'], x['snippet'], x['link']]) for x in results]
    )
    
    
    # Return the formatted results or a 'No results found.' message if no results exist.
    return formatted_results if results else 'No results found.'


@tool(args_schema=ArxivInput) # Use the Pydantic schema
def fetch_arxiv(arvix_id: str) -> str:
    '''fetch arXiv abstract'''
    print('New fetch arXiv abstract function')
    # 1. Fetch the content
    # Note: Older arXiv IDs like 9308101 might redirect or have simpler pages.
    res = requests.get(f'https://arxiv.org/abs/{arvix_id}')
    
    # 2. Safely attempt the regex search
    match = abstract_pattern.search(res.text)
    
    if match:
        # If the expected blockquote structure is found, extract and return
        abstract = match.group(1).strip()
        if abstract:
            return abstract
        else:
            # Found the block, but it was empty
            raise ValueError(f'Found empty abstract block for {arvix_id}')
    
    # 3. Fallback/Error handling: Try a more generic match for older pages
    # Older pages often have the abstract text directly in a div or p tag.
    # This is an example of checking for a different, simpler pattern.
    fallback_pattern = re.compile(
        r'<div id="abstract">\s*<blockquote.*?>(.*?)</blockquote>',
        re.DOTALL
    )
    fallback_match = fallback_pattern.search(res.text)
    
    if fallback_match:
        return fallback_match.group(1).strip()
    
    # If no pattern matches, the error is legitimate
    raise ValueError(f'Could not find abstract content for arXiv ID: {arvix_id}. The page structure may be non-standard.')    