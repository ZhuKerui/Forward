import json
import networkx as nx
from typing import List
import math
from tools.TextProcessing import sent_lemmatize
from tools.BasicUtils import my_json_read, my_read

class SentenceReformer:
    
    def __init__(self, json_file:str):
        self.MyTree = json.load(open(json_file, 'r'))
        self.line_record = []

    def line_operation(self, line:str):
        if not line:
            return
        word_tokens = sent_lemmatize(line)
        reformed_sent = []
        i = 0
        while i < len(word_tokens):
            if word_tokens[i] not in self.MyTree.keys():
                # If the word is not part of a key word directly add the word to the reformed_sent list
                reformed_sent.append(word_tokens[i])
                i += 1

            else:
                # If the word is part of a key word
                phrase_buf = []
                phrase_wait_buf = []
                tail_buf = []
                it = self.MyTree
                while i < len(word_tokens) and word_tokens[i] in it.keys():
                    # Add the word to the wait list
                    phrase_wait_buf.append(word_tokens[i])
                    tail_buf.append(word_tokens[i])
                    if "" in it[word_tokens[i]].keys():
                        # If the word could be the last word of a keyword, update the phrase buffer to be the same with wait buffer
                        phrase_buf = phrase_wait_buf.copy()
                        tail_buf = []
                    # Go down the tree to the next child
                    it = it[word_tokens[i]]
                    i += 1
                # Change the keyword into one uniformed word and add it to the reformed_sent
                if phrase_buf:
                    reformed_sent.append('_'.join(phrase_buf))
                reformed_sent += tail_buf
        self.line_record.append(' '.join(reformed_sent))


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

def co_occur_load(co_occur_file:str):
    return [line.split('\t') for line in my_read(co_occur_file)]


def occurrence_load(occur_file:str):
    with open(occur_file) as f_in:
        temp_dict = json.load(f_in)
        return {str(key):set(val) for key, val in temp_dict.items()}

# Graph related functions
def build_graph(co_occur_list:List[List[str]], keyword_list:List[str]):
    g = nx.Graph(c=0)
    g.add_nodes_from(keyword_list, c=0)
    print('Reading Co-occurrence lines')
    for line_idx, line in enumerate(co_occur_list):
        kw_num = len(line)
        g.graph['c'] += kw_num * (kw_num - 1)
        for i in range(kw_num):
            u = line[i]
            g.nodes[u]['c'] += (kw_num - 1)
            for j in range(i+1, kw_num):
                v = line[j]
                if not g.has_edge(u, v):
                    g.add_edge(u, v, c=0)
                g.edges[u, v]['c'] += 1
        if line_idx % 5000 == 0:
            print('\r%d' % line_idx, end='')
    print('')
    print('Reading Done! NPMI analysis starts...')
    Z = float(g.graph['c'])
    for e, attr in g.edges.items():
        attr['npmi'] = -math.log((2 * Z * attr['c']) / (g.nodes[e[0]]['c'] * g.nodes[e[1]]['c'])) / math.log(2 * attr['c'] / Z)
    print('NPMI analysis Done')
    return g

def graph_dump(g:nx.Graph, gpickle_file:str):
    nx.write_gpickle(g, gpickle_file)

def graph_load(gpickle_file:str):
    return nx.read_gpickle(gpickle_file)

def get_subgraph(g:nx.Graph, npmi_min:float, min_count:int, npmi_max:float=1.0):
    return g.edge_subgraph([e[0] for e in g.edges.items() if e[1]['npmi'] > npmi_min and e[1]['c'] >= min_count and e[1]['npmi'] < npmi_max])
