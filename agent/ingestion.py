import requests
import pandas as pd
import json
import xml.etree.ElementTree as ET
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv(dotenv_path="./cred.env")


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

encoder = OpenAIEncoder(name='text-embedding-3-small')


pc = Pinecone(api_key=PINECONE_API_KEY)

##define serveless specification
spec = ServerlessSpec(
        cloud='aws',
        region='us-east-1'
)

index_name = 'langgraph-research-agent'

ARXIV_NAMESPACE = '{http://www.w3.org/2005/Atom}'


def extract_from_arxiv (search_query='cat:cs.AI', max_results=50, json_file_path='files/arxiv_dataset.json'):
    """
    Search papers from ARXIV API and save them as JSON

    Args:
        search_query (str): The search query for ArXiv (default is 'cat:cs.AI').
        max_results (int): The maximum number of results to retrieve (default is 100).
        json_file_path (str): File path where JSON data will be saved.

    Returns:
        pd.DataFrame: DataFrame containing the extracted paper information.

    """


## check documentation at https://info.arxiv.org/help/api/user-manual.html#412-python

    url = f'http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}'

    #http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=50

    #response = requests.get(url)
    #print(response.text)

    # with open('files/old_response.txt', 'r', encoding='utf-8') as f:
    #     f.read(old_response)

    old_reponse_file = 'files/old_response.xml'

    #root = ET.fromstring(old_reponse_file)
    tree = ET.parse(old_reponse_file)
    root = tree.getroot()
    print(type(root))

    papers=[]

    ## find all for multiple elements and find for first single element find
    for entry in root.findall(f'{ARXIV_NAMESPACE}entry'):
        title = entry.find(f'{ARXIV_NAMESPACE}title').text.strip()
        summary = entry.find(f'{ARXIV_NAMESPACE}summary').text.strip()

        #Get all authors
        author_elements= entry.findall(f'{ARXIV_NAMESPACE}author')
        authors = [ authors.find(f'{ARXIV_NAMESPACE}name').text    for authors in author_elements]
        #print(f'authors: {authors}')

        #get paper url
        url = entry.find(f'{ARXIV_NAMESPACE}id').text.strip()
        #print(f'url: {url} \n')

        arxiv_id = url.split('/')[-1]
        #print(f'arxiv_id: {arxiv_id} \n')

        ##check for pdf link
        pdf_link_element = entry.find(f'{ARXIV_NAMESPACE}link[@title="pdf"]')
        if pdf_link_element is not None:
            pdf_link = pdf_link_element.attrib.get('href')
            print(f'pdf_link: {pdf_link} \n')
        else:
            pdf_link = None
            print(f'pdf_link NOT found: {pdf_link} \n')


        # pdf_link = entry.find(f'{ARXIV_NAMESPACE}link[@title="pdf"]').attrib.get('href')
        # #print(f'pdf_link: {pdf_link} \n')

        papers.append({
            'title': title,
            'summary': summary,
            'authors': authors,
            'arxiv_id': arxiv_id,
            'url': url,
            'pdf_link': pdf_link
        })

    df = pd.DataFrame(papers)
    

    # Save the DataFrame to a JSON file.
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)
        print(f'Data saved to {json_file_path} ...')

    return df



def download_pdfs(df, download_folder='files'):
    '''
    Download PDF from df and save it in local folder
    '''
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    pdf_file_names = [] ## empty list for storing pdf file names
    
    for index, row in df.iterrows():
        pdf_link = row['pdf_link']

        try:
            response = requests.get(pdf_link)
            response.raise_for_status()

            file_name = os.path.join(download_folder, pdf_link.split('/')[-1]) + '.pdf'
            pdf_file_names.append(file_name)

            # Save the downloaded PDF
            # with open(file_name, 'wb') as f:
            #     f.write(response.content)
            
            #print(f'PDF downloaded successfully and saved as {file_name}')
        
        except requests.exceptions.RequestException as e:
            print(f'Failed to download the PDF: {e}')
            pdf_file_names.append(None)
    
    df['pdf_file_name'] = pdf_file_names

    return df


def load_pdf_chunks(file_path):
    '''Load pdf file and chunk it'''
    loader = PyPDFLoader(file_path)
    data= loader.load()
    
    # Initialize the RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Maximum size of each chunk
        chunk_overlap=50  # Overlap between chunks
    )

    # Split the text
    chunks = splitter.split_documents(data)

    return chunks



def expand_df(df):
    '''Expand DF to chunks
    return New expanded df
    '''

    expanded_row = []

    ## loop through each row in DF
    for index,row in df.iterrows():
        file_name = row['pdf_file_name']
        #print(row)
        try:
            chunks = load_pdf_chunks(file_name)
        except:
            print(f'PDF file not found for {index} row with filename - {file_name}')
            continue

        #loop over the chunks and add it to new data frame
        #print(f'Adding {len(chunks)} chunks for {index} row')
        for i, chunk in enumerate(chunks):
            prechunk = i-1 if i > 0 else '' # Preceding chunk ID
            postchunk = i+1 if i < len(chunks) -1 else '' # Following chunk ID

            expanded_row.append(
                {
                    'id':"{}#{}".format(row['arxiv_id'],i),
                    'title':row['title'],
                    'summary':row['summary'],
                    'authors':row['authors'],
                    'arxiv_id':row['arxiv_id'],
                    'url':row['url'],
                    'chunk':chunk.page_content,
                    'prechunk_id': '' if i == 0 else "{}#{}".format(row['arxiv_id'],prechunk),
                    'postchunk_id': '' if i == len(chunks) -1 else "{}#{}".format(row['arxiv_id'],postchunk)


                }
            )

    return pd.DataFrame(expanded_row)        

def get_pinecone_index(index_name: str):
    '''Get or create Pinecone index'''
    if index_name not in pc.list_indexes():
        print(f'Creating index: {index_name}')
        pc.create_index(
            name=index_name,
            dimension=1536, # Dimension for OpenAI text-embedding-3-small
            metric='cosine',
            serverless_spec=spec
        )
        # Wait for a few seconds to ensure the index is ready
        time.sleep(10)
    else:
        print(f'Index {index_name} already exists')
    
    index = pc.get_index(index_name)
    return index

#expand_df(df)


def approx_payload_size_bytes(batch_tuples):
    """
    Approximate JSON-serialized size of the upsert payload.
    This is conservative but good for dynamic batching.
    """
    # Represent as list of dicts similar to Pinecone upsert format
    sample = []
    for id, vec, meta in batch_tuples:
        sample.append({"id": id, "values": vec, "metadata": meta})
    return len(json.dumps(sample).encode('utf-8'))/(1024 * 1024)



def upsert_data(data, index, batch_size=64):
    '''Upsert data to Pinecone'''

    for i in tqdm(range(0,len(data),batch_size)):
        print("iteration:",i)
        i_end = min(len(data),i+batch_size) ##end point

        #print("i_end:",i_end)

        batch = data[i:i_end].to_dict(orient='records')

        #print("len(batch):",len(batch))

        ##get metadata and ID for each chunk in batch

        metadata= [{'arxiv_id':r['arxiv_id'],'title':r['title'],'chunk':r['chunk']} for r in batch]

        ids = [r['id'] for r in batch]

        chunks = [r['chunk'] for r in batch]

        print("len(chunks) sample:",len(chunks))
        
        embeds = encoder(chunks)## openai encoder function isntead of tiktoken



        batch_tuple = [(ids[j],embeds[j],metadata[j]) for j in range(len(ids))]

        #print("Current batch size:", len(batch_tuple))
        #print("First 3 chunk types:", [type(c) for c in chunks[:3]])
        #print("Any None in chunks?:", any(c is None for c in chunks))

        payload_size = approx_payload_size_bytes(batch_tuple)
        print(f'Payload size: {payload_size} MBs')

        ##upload embeddings , ids and metadata
        index.upsert(vectors=zip(ids,embeds,metadata))


if __name__ == '__main__':
    #testing the ingestion pipeline
    df = extract_from_arxiv()

    df = download_pdfs(df)
    expanded_df = expand_df(df)
    index = get_pinecone_index(index_name=index_name)
    upsert_data(expanded_df, index=index, batch_size=16)

       