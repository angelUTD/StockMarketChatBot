import requests
import os
import argparse
import json
from dotenv import load_dotenv
from readability import Document
from bs4 import BeautifulSoup
import pandas as pd
from transformers import GPT2Tokenizer
import re

# define utility functions

# Create a tuple of (title, section_name, content, number of tokens) for each section in the html document.
def section(html, title):
    # split by h2 tags
    # insert a new h2 tag as first element of body
    # this is to ensure that the first section is included
    initSection = html.new_tag('h2')
    initSection.string = title
    html.body.insert(0, initSection)
    # get all h2 tags
    sections = html.find_all('h2')
    section_list = []
    for s in sections:
        # get section name
        name = s.get_text()
        #store all content between h2 tags
        content = s.next_sibling
        contentText = content.get_text() if content else ''
        # loop through siblings until next h2 tag is found
        if content:
            while content.next_sibling != None and content.next_sibling.name != 'h2':
                # append content to contentText
                contentText += content.next_sibling.get_text().strip()
                # move to next sibling
                content = content.next_sibling

        contentText = re.sub('[\n]+', '\n', contentText)
        # remove leading and trailing whitespace
        contentText = contentText.strip()

        # get number of tokens
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = len(tokenizer.encode(contentText, truncation=True))
        # create tuple
        section = (title, name, contentText, tokens)
        # append to list if nontrivial
        if tokens > 0:
            section_list.append(section)
    return section_list

# define script arguments
parser = argparse.ArgumentParser(description='Scrape search results from Google Custom Search API')
parser.add_argument('--query', '-q', type=str, help='Search query', default='What is machine learning?')
parser.add_argument('--debug', '-d', action='store_true', help='Debug mode (stores outputs, prints debug statements)', default=False)
parser.add_argument('--output', '-o', type=str, help='Output file name', default='search_results.csv')
parser.add_argument('--num_results', '-n', type=int, help='Number of top results to scrape', default=3, choices=range(1,11))
args = parser.parse_args()

# set environment variables
load_dotenv('api.env')
# set API key
API_KEY = os.getenv("GOOGLE_API_KEY")
# end program if environment variables are not set
if not API_KEY:
    raise RuntimeError("API_KEY not set")

# Change working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# set other parameters
query = args.query
cx = "234c926134b3e4c93"

# set up API search request
url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={cx}&q={query}'
response = requests.get(url)
#print('url:', url)
#print(response.text)
# check if request was successful
if response.status_code == 200:
    # save to JSON file
    if args.debug:
        with open('search_results.json', 'w') as f:
            json.dump(response.json(), f, indent=4)
        # print links
    results = []
    for x, item in enumerate(response.json()['items'][0:args.num_results]):
        link = item['link']
        if args.debug: print(link)
        # get page content
        page = requests.get(link)
        doc = Document(page.content, min_text_length=200)
        # save pages to HTML files
        # sanitize title for file name
        title = "".join(x for x in doc.short_title() if x.isalnum() or x in " _-")
        if args.debug:
            with open(f'debug_pages/{x} {title}.html', 'w', encoding='utf-8') as f:
                f.write(doc.summary())
        # get sections
        html = BeautifulSoup(str(doc.summary()), 'html.parser')
        sections = section(html, title)
        # print sections
        for s in sections:
            body = s[2]
            if len(body) > 100:
                body = body[0:100] + '...'
            if args.debug: print(f'{s[0]} > {s[1]} {body} {s[3]} tokens\n')
            # append to results
            results.append(s)

    # save results to CSV
    with open(args.output, 'a') as f:
        df = pd.DataFrame(results, columns=['title', 'section', 'content', 'tokens'])
        df.to_csv(f, header=f.tell()==0, index=False)
        

