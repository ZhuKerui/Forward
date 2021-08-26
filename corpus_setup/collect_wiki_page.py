# python collect_wiki_page.py wikipedia_ent.txt wiki_pages.json

import json
import sys
import wikipedia
import time

sys.path.append('..')
from tools.BasicUtils import my_read, MultiThreading

start_time = time.time()
wiki_ent_list = my_read(sys.argv[1])
def get_wikipedia_json(ent:str):
    try:
        p = wikipedia.WikipediaPage(ent)
        d = {sec : p.section(sec) for sec in p.sections}
        d['summary'] = p.summary
        d['title'] = p.title
        return d
    except:
        return None
mt = MultiThreading()
ret = mt.run(get_wikipedia_json, wiki_ent_list, 10)
ret = [item for item in ret if item is not None]
json.dump(ret, open(sys.argv[2], 'w'))
print(time.time() - start_time)