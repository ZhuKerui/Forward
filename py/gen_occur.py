# python gen_occur.py keyword_file co_occur_file occur_file
import tqdm
import pickle
import sys

sys.path.append('..')

from tools.BasicUtils import my_read
from tools.DocProcessing import co_occur_load

kws = my_read(sys.argv[1])
co_occur = co_occur_load(sys.argv[2])
occur_dict = {kw:set() for kw in kws}
for i, co_kws in enumerate(tqdm.tqdm(co_occur)):
    if not co_kws:
        continue
    for kw in co_kws:
        occur_dict[kw].add(i)
with open(sys.argv[3], 'wb') as f_out:
    pickle.dump(occur_dict, f_out)
