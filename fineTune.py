import subprocess 
import os
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import time
import pickle 
import tiktoken



MAX_SECTION_LEN = 1000  
MAX_COMPLETION_TOKENS = 300 
MAX_CONTENT_TOKENS = 500 
DL_PATH = 'search_results.csv'
COMPLETIONS_MODEL = "text-davinci-003"  # model used to answer the query
EMBEDDING_MODEL = "text-embedding-ada-002"  # model used to produce embeddings
preprompt = 'Answer the question as truthfully as possible.'
EMBEDS_PATH = 'embeds.csv'
SEPARATOR = "\n* "
ENCODING = "gpt2"
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": MAX_COMPLETION_TOKENS,
    "model": COMPLETIONS_MODEL,
}
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))



def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = True
) -> str:
    # construct prompt from query, embeds, and document library
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    # show the prompt if option is enabled
    # feed the prompt into the OpenAI API to get the completion
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    # get ordered list of relevant documents to query
    most_relevant_document_sections = order_document_sections_by_query_similarity(df, question, context_embeddings)
    # accumulators for sections we want to inclde in the prompt
    chosen_sections = []    # section strings preceded by separator
    chosen_sections_len = 0 # token count for all sections- includes separator token count
    chosen_sections_indexes = []    # (title, section) of articles used


    # add contexts to prompt until token limit is reached  

    for _, section_index in most_relevant_document_sections:      
        print(section_index)
        document_section = df.loc[section_index] # get (title, section) 
        temp = document_section.tokens
        chosen_sections_len += document_section.tokens + separator_len  # accumulate the no. tokens from sections
        if chosen_sections_len > MAX_SECTION_LEN.any():
            break
        # add separators between sections
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    
    # pre-prompt 
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = """Answer the question as truthfully as possible using the provided context and existing knowledge, and if you are not confident in your answer, say "I don't know."\n\nContext:\n"""

    #header = 'Answer the question using the provided context.\n\n Context:\n'
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(dl, query: str, contexts: dict[(str, str), np.array], max_tokens=MAX_CONTENT_TOKENS) -> list[(float, (str, str))]:
    # get embedding for the query
    query_embedding = get_embedding(query) # returns a dict
    
    # compare the query embedding against document embeddings in dl -> returns a list of these structs: (similarity_score, (title, section))
    document_similarities = sorted([
        # doc_index is the (title, section) tuple
        # vector_similarity() returns a value from 0-1 that reflects the similarity to the query
        (vector_similarity(query_embedding, doc_embedding), doc_index) 
        for doc_index, doc_embedding in contexts.items() if (dl.loc[doc_index]['tokens'] <= max_tokens).any()
    ])

    return document_similarities
    

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "section"])
    return {
           (r.title, r.section): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    embeds_dict = {
        idx: get_embedding(r.content) for idx, r in df.iloc[::15].iterrows()
    }
    #print(embeds_dict)
    # append new embeds_dict to existing .csv
    append_embeds(embeds_dict)

    return embeds_dict


def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result['data'][0]['embedding']


def append_embeds(input_dict: dict):
    with open(EMBEDS_PATH, 'w', encoding='latin1') as f:
        # convert new embeds_dict into dataframe
        data_list = []
        for key, value in input_dict.items():
            data_dict = {"title": key[0], "section": key[1]}
            for i, v in enumerate(value):
                data_dict[i] = v
            data_list.append(data_dict)
        df = pd.DataFrame(data_list)
        df.to_csv(f, header=f.tell()==0, index=False, encoding='latin1')
        print(f'Appended new embeds to {EMBEDS_PATH}.')
    # clean .csv   
    remove_duplicates_from_csv(EMBEDS_PATH)

def init_api_key():
    load_dotenv('api.env')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise RuntimeError("API_KEY not set")

def remove_duplicates_from_csv(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.drop_duplicates()
    df.to_csv(file_path, index=False)

def tune(query):
    with open('api.env') as f:
        env_vars = dict(line.strip().split('=') for line in f)
    # Add the environment variables to the subprocess environment
    subprocess_env = os.environ.copy()
    subprocess_env.update(env_vars)

    # get top n articles and store in search_results.csv
    num_articles = 10
    cmd = ['python', 'search_scraper.py', '-q', query, '-n', str(num_articles)]
    #print(cmd)
    try:
        result = subprocess.run(cmd, shell=True, env=subprocess_env, stdout=subprocess.PIPE,  stderr=subprocess.PIPE, check=True)
    except: 
        print("error")
    
    remove_duplicates_from_csv(DL_PATH)

    dl = pd.read_csv(DL_PATH, encoding='latin1')
    dl = dl.set_index(["title", "section"])

    init_api_key()

    dl = pd.read_csv('search_results.csv')
    dl = dl.set_index(['title', 'section'])

    old_embeds_dict = {}
    if os.path.exists(EMBEDS_PATH):
        try:
            old_embeds_dict = load_embeddings(EMBEDS_PATH)
        except:
            x=2
    else:
        x=1


    init_api_key()  # load API key if not already loaded
    embeds_dict = compute_doc_embeddings(dl) # FIX HERE

    embeds = {**old_embeds_dict, **embeds_dict}

    remove_duplicates_from_csv(EMBEDS_PATH)

    docSim = order_document_sections_by_query_similarity(dl, query, embeds)[:5]

    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))

    answer = answer_query_with_context(query, dl, embeds)
    return(answer)
