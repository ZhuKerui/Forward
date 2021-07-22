# python collect_wiki_page.py kw_ent_map.json wiki_pages.json

import json
import sys
import wikipedia
import time

sys.path.append('..')
from tools.BasicUtils import my_json_read, MultiThreading

start_time = time.time()
kw2ent_dict = my_json_read(sys.argv[1])
wiki_ent_list = list(set(kw2ent_dict.values()))
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
json.dump(ret, open(sys.argv[2], 'w'))
print(time.time() - start_time)