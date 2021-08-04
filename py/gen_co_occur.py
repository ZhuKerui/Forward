# python gen_co_occur.py wordtree_file sentence_file co_occur_file
import tqdm
import sys

sys.path.append('..')

from tools.BasicUtils import my_json_read, my_read, my_write
from tools.TextProcessing import sent_lemmatize

class CoOccurrence:
    def __init__(self, wordtree_file:str):
        self.wordtree = my_json_read(wordtree_file)

    def line_operation(self, reformed_sent:list):
        i = 0
        kw_set_for_line = set()
        while i < len(reformed_sent):
            if reformed_sent[i] in self.wordtree: # If the word is the start word of a keyword
                phrase_buf = []
                it = self.wordtree
                j = i
                while j < len(reformed_sent) and reformed_sent[j] in it:
                    # Add the word to the wait list
                    phrase_buf.append(reformed_sent[j])
                    if "" in it[reformed_sent[j]]: # If the word could be the last word of a keyword, update the list
                        kw_set_for_line.add(' '.join(phrase_buf).replace(' - ', '-'))
                    # Go down the tree to the next child
                    it = it[reformed_sent[j]]
                    j += 1
                    i = j - 1
            i += 1
        return kw_set_for_line if kw_set_for_line else None

if __name__ == '__main__':
    o = CoOccurrence(sys.argv[1])
    sents = my_read(sys.argv[2])
    ret = [o.line_operation(sent_lemmatize(sent)) for sent in tqdm.tqdm(sents)]
    ret_text = ['' if item is None else '\t'.join(item) for item in ret]
    my_write(sys.argv[3], ret_text)
