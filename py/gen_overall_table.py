# python gen_overall_table.py entity_file co_occur_file pair_graph_file sentence_file dataset_file
import sys
sys.path.append('..')
from tools.DocProcessing import co_occur_load
from tools.BasicUtils import my_read, my_email
from tools.DocProcessing import graph_load
from tools.TextProcessing import nlp, find_dependency_path_from_tree
import pandas as pd
import datetime
from tqdm import tqdm

start_time = str(datetime.datetime.now())

keyword_list = my_read(sys.argv[1])
co_occur_list = co_occur_load(sys.argv[2])
pair_graph = graph_load(sys.argv[3])

print('Data loaded, start running...')

df = pd.DataFrame()
data = []

for idx, line in tqdm(enumerate(open(sys.argv[4]).readlines())):
    tokens = line.split()
    kws = co_occur_list[idx]
    kws = [kw_idx for kw_idx in kws if tokens.count(keyword_list[kw_idx]) == 1]
    if len(kws) <= 1:
        continue
    sent_len = len(tokens)
    pairs = [(kws[i], kws[j]) for i in range(len(kws)-1) for j in range(i+1, len(kws))]
    doc = nlp(line)
    for kw_1, kw_2 in pairs:
        npmi = pair_graph.edges[kw_1, kw_2]['npmi']
        kw_1_text, kw_2_text = keyword_list[kw_1], keyword_list[kw_2]
        kw_dist = abs(tokens.index(kw_1_text) - tokens.index(kw_2_text))
        path_1 = find_dependency_path_from_tree(doc, kw_1_text, kw_2_text)
        path_2 = ' '.join([token[2:] if token[:2] == 'i_' else 'i_' + token for token in path_1.split()]) if path_1 else ''
        data.append((idx, sent_len, kw_1, kw_2, path_1, npmi, kw_dist))
        data.append((idx, sent_len, kw_1, kw_2, path_2, npmi, kw_dist))
pd.DataFrame(data, columns=['sent_id', 'sent_len', 'head_kw_id', 'tail_kw_id', 'dep_path', 'npmi', 'kw_dist']).to_csv(sys.argv[5], index=False)

end_time = str(datetime.datetime.now())
message = "Task is done.\nStarting time: %s ;\nEnd time: %s" % (start_time, end_time)
my_email("Task done", message, "keruiz2@illinois.edu")