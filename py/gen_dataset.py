# python gen_dataset.py entity_file co_occur_file pair_graph_file idx_sentence_file dataset_file
import sys
sys.path.append('..')
from tools.DocProcessing.DatasetGenerator import Dataset_Generator, dataset_generator_post_operation
from tools.DocProcessing.CoOccurrence import co_occur_load
from tools.BasicUtils import MultiProcessing, my_read, my_email
from tools.DocProcessing.CoOccurGraph import graph_load, get_subgraph
import time
import pandas as pd
import datetime

start_time = str(datetime.datetime.now())

# p = MultiProcessing()
keyword_list = my_read(sys.argv[1])
co_occur_list = co_occur_load(sys.argv[2])
pair_graph = get_subgraph(graph_load(sys.argv[3]), 0.3, 3)
print('Data loaded, start running...')
# df = p.run(lambda: Dataset_Generator(keyword_list, co_occur_list, pair_graph), open(sys.argv[4]).readlines(), 8, dataset_generator_post_operation)
# df.to_csv(sys.argv[5])
dg = Dataset_Generator(keyword_list, co_occur_list, pair_graph)
for idx, line in enumerate(open(sys.argv[4]).readlines()):
    dg.line_operation(line)
print('Use time %f seconds' % (time.time() - start_time))
pd.DataFrame(dg.line_record, columns=['path', 'subj', 'obj']).to_csv(sys.argv[5], index=False)

end_time = str(datetime.datetime.now())
message = "Task is done.\nStarting time: %s ;\nEnd time: %s" % (start_time, end_time)
my_email("Task done", message, "keruiz2@illinois.edu")