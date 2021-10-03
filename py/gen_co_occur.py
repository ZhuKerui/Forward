# python gen_co_occur.py wordtree_file token_file use_greedy(T/F) sentence_file co_occur_file
import tqdm
import sys

sys.path.append('..')

from tools.BasicUtils import my_read, my_write
from tools.TextProcessing import sent_lemmatize
from tools.DocProcessing import CoOccurrence

o = CoOccurrence(sys.argv[1], sys.argv[2], sys.argv[3] == 'T')
sents = my_read(sys.argv[4])
ret_text = ['\t'.join(o.line_operation(sent_lemmatize(sent))) for sent in tqdm.tqdm(sents)]
my_write(sys.argv[5], ret_text)
