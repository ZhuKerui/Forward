# python gen_occur.py keyword_file co_occur_file occur_file
import tqdm
import json
import sys

sys.path.append('..')

from tools.BasicUtils import my_read

kws = my_read(sys.argv[1])
co_occur = my_read(sys.argv[2])
occur_dict = {kw:[] for kw in kws}
for i, co_kws in tqdm.tqdm(enumerate(co_occur)):
    if co_kws == '':
        continue
    co_kws = co_kws.split('\t')
    for kw in co_kws:
        occur_dict[kw].append(i)
json.dump(occur_dict, open(sys.argv[3], 'w'))
