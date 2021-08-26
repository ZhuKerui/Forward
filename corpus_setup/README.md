# corpus_setup folder

> This folder provides useful tools for processing the raw data under "data/raw_data" folder to more task-specific dataset under the "data/corpus" folder

## Process and select keywords

- **keyword_process.ipynb**
    - Select cs keywords from terms-cs-cfl-epoch200.txt
    - Find wikipedia entity for each keyword and build a map
    - Collect cs wikipedia entity

## Collect wikipedia pages for keywords

- **collect_wiki_page.py**

## Get sentences from arxiv abstraction

- **get_sent_from_arxiv.ipynb**
    - Grab sentences from arxiv abstraction and implement sentence tokenization
    - For both big_arxiv.json and small_arxiv.json

## Get openie triples

- **openie2dataset.ipynb**
    - Filter sentences from TupleInfKB which can be decomposed to triples