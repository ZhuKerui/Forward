# python filter_by_path.py wordtree_file sentence_file output_file use_id
import tqdm
import re
import pandas as pd
import sys

sys.path.append('..')

from tools.BasicUtils import my_json_read, my_read
from tools.TextProcessing import sent_lemmatize, nlp, find_dependency_path_from_tree, find_span


def exact_match(pattern:re.Pattern, path:str):
    mat = pattern.match(path)
    if mat is None:
        return False
    return len(path) == mat.end()

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
    patterns = [
        'i_nsubj attr( prep pobj)*', 
        'i_nsubj( conj)* dobj( acl prep pobj( conj)*){0,1}', 
        'i_nsubj( prep pobj)+( conj)*', 
        'i_nsubj advcl dobj( acl attr){0,1}', 
        'appos( conj)*', 
        'appos acl prep pobj( conj)*', 
        'i_nsubjpass( conj)*( prep pobj)+( conj)*', 
        'i_nsubjpass prep pobj acl dobj', 
        'i_dobj prep pobj( conj)*', 
        'acl prep pobj( conj)*'
    ]
    matcher = re.compile('|'.join(patterns))

    o = CoOccurrence(sys.argv[1])
    sents = my_read(sys.argv[2])
    use_id = sys.argv[4].lower() == 'true'

    df = pd.DataFrame(columns=['head', 'head_span', 'tail', 'tail_span', 'sent', 'path'])

    for idx, sent in enumerate(tqdm.tqdm(sents)):
        kws = o.line_operation(sent_lemmatize(sent))
        if kws is None or len(kws) < 2:
            continue
        kws = list(kws)
        doc = nlp(sent)
        spans = [find_span(doc, kw, True) for kw in kws]
        for i in range(len(spans)-1):
            for i_ in spans[i]:
                for j in range(1, len(spans)):
                    for j_ in spans[j]:
                        path = find_dependency_path_from_tree(doc, doc[i_[0]:i_[1]], doc[j_[0]:j_[1]])
                        if not path:
                            continue
                        i_path = [token[2:] if token[:2] == 'i_' else 'i_' + token for token in path.split()]
                        i_path.reverse()
                        i_path = ' '.join(i_path)
                        if exact_match(matcher, path):
                            df = df.append({'head':kws[i],
                                    'head_span':i_,
                                    'tail':kws[j],
                                    'tail_span':j_,
                                    'sent':sent if not use_id else idx,
                                    'path':path}, ignore_index=True)
                        if exact_match(matcher, i_path):
                            df = df.append({'head':kws[j],
                                    'head_span':j_,
                                    'tail':kws[i],
                                    'tail_span':i_,
                                    'sent':sent if not use_id else idx,
                                    'path':i_path}, ignore_index=True)
    df.to_csv(sys.argv[3], sep='\t', index=False)
