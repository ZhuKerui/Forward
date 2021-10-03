# python extract_wiki.py collect_sents
# python extract_wiki.py collect_sents sentence_file output_file use_id[T/F] keyword_only[T/F]
# python extract_wiki.py collect_kw_occur_from_selected
# python extract_wiki.py build_graph

import re
import os
import tqdm
import csv
import pickle
import networkx as nx
import pandas as pd
from urllib.parse import unquote
from typing import List
import sys
sys.path.append('..')

from tools.BasicUtils import MyMultiProcessing, my_read, my_write
from tools.TextProcessing import remove_brackets, batched_sent_tokenize, nlp, find_span, sent_lemmatize, find_noun_phrases, find_dependency_path_from_tree, exact_match
from tools.DocProcessing import CoOccurrence


# Some constants
wikipedia_dir = '../../data/wikipedia/full_text-2021-03-20'
wikipedia_entity_file = 'data/extract_wiki/wikipedia_entity.tsv'
wikipedia_entity_norm_file = 'data/extract_wiki/wikipedia_entity_norm.tsv'
wikipedia_keyword_file = 'data/extract_wiki/keywords.txt'
wikipedia_wordtree_file = 'data/extract_wiki/wordtree.pickle'
wikipedia_token_file = 'data/extract_wiki/tokens.txt'
save_path = 'data/extract_wiki/wiki_sent_collect'
keyword_occur_file = 'data/extract_wiki/keyword_occur.pickle'
keyword_connection_graph_file = 'data/extract_wiki/keyword_graph.pickle'


# Some task specific classes
class SentenceFilter:
    def __init__(self, wordtree_file:str=None, token_file:str=None):
        if wordtree_file and token_file:
            self.co_occur_finder = CoOccurrence(wordtree_file, token_file)
        else:
            self.co_occur_finder = None
        patterns = [
            'i_nsubj attr( prep pobj)*( compound){0, 1}', 
            'i_nsubj( conj)* dobj( acl prep pobj( conj)*){0,1}', 
            'i_nsubj( prep pobj)+( conj)*', 
            'i_nsubj advcl dobj( acl attr){0,1}', 
            'appos( conj)*', 
            'appos acl prep pobj( conj)*', 
            'i_nsubjpass( conj)*( prep pobj)+( conj)*', 
            'i_nsubjpass prep pobj acl dobj', 
            'i_dobj prep pobj( conj)*'
            # 'acl prep pobj( conj)*'
        ]
        self.matcher = re.compile('|'.join(patterns))

    def list_operation(self, sents:List[str], use_id:bool=False, keyword_only:bool=False):
        """
        Select sentences that contain specific dependency path

        Inputs
        ----------
        sents : List[str]
            a list of sentences to be selected

        use_id : bool
            use the id of the sentence in the list or not to record the sentence

        keyword_only : bool
            entities are keywords in the sentence if true, otherwise, just any noun phrases

        Return
        -------
        A pandas.Dataframe containing columns: ['head', 'head_norm', 'head_span', 'tail', 'tail_norm', 'tail_span', 'sent', 'path']
        """

        df = pd.DataFrame(columns=['head', 'head_norm', 'head_span', 'tail', 'tail_norm', 'tail_span', 'sent', 'path'])

        for idx, sent in enumerate(sents):
            doc = nlp(sent)
            if keyword_only and self.co_occur_finder:
                kws_set = self.co_occur_finder.line_operation(sent_lemmatize(sent))
                kws = []
                kws_norm = []
                for kw in kws_set:
                    spans = find_span(doc, kw, use_lemma=True)
                    kws += spans
                    kws_norm += [kw] * len(spans)
            else:
                kws = find_noun_phrases(doc)
            if len(kws) < 2:
                continue
            for i in range(len(kws)-1):
                for j in range(i, len(kws)):
                    path = find_dependency_path_from_tree(doc, kws[i], kws[j])
                    if not path:
                        continue
                    i_path = [token[2:] if token[:2] == 'i_' else 'i_' + token for token in path.split()]
                    i_path.reverse()
                    i_path = ' '.join(i_path)
                    if exact_match(self.matcher, path):
                        head_idx, tail_idx = i, j
                    elif exact_match(self.matcher, i_path):
                        head_idx, tail_idx = j, i
                    else:
                        continue
                    new_data = {'head':kws[head_idx].text,
                                'head_span':(kws[head_idx][0].i, kws[head_idx][-1].i),
                                'tail':kws[tail_idx].text,
                                'tail_span':(kws[tail_idx][0].i, kws[tail_idx][-1].i),
                                'sent':sent if not use_id else idx,
                                'path':path}
                    if keyword_only:
                        new_data['head_norm'], new_data['tail_norm'] = kws_norm[head_idx], kws_norm[tail_idx]
                    else:
                        new_data['head_norm'], new_data['tail_norm'] = ' '.join(sent_lemmatize(kws[head_idx].text)), ' '.join(sent_lemmatize(kws[tail_idx].text))
                    df = df.append(new_data, ignore_index=True)
        return df


# Some helper functions
def collect_wiki_entity(file:str):
    return ['%s\t%s' % (line[9:re.search(r'" url="', line).start()], line[re.search(r'title="', line).end():re.search(r'">', line).start()]) for line in open(file).readlines() if re.match(r'^(<doc id=")', line) and line.isascii()]


def get_sentence(wiki_file:str, save_sent_file:str):
    paragraphs = []
    with open(wiki_file) as f_in:
        # Remove useless lines
        entity_name = ''
        for line in f_in:
            line = line.strip()
            if not line or line == entity_name:
                continue
            if re.match(r'^(<doc id=")', line):
                entity_name = line[re.search(r'title="', line).end():re.search(r'">', line).start()]
                # entity_id = line[9:re.search(r'" url="', line).start()]
                # keyword = remove_brackets(normalize_text(entity_name))
            else:
                # Process the links in the paragraph
                links = re.findall(r'<a href="[^"]*">[^<]*</a>', line)
                for l in links:
                    breakpoint = l.index('">')
                    ent = remove_brackets(unquote(l[9:breakpoint])).lower()
                    kw = l[breakpoint+2:-4].lower()
                    if ent[-len(kw):] == kw:
                        line = line.replace(l, ent)
                    else:
                        line = line.replace(l, kw)
                paragraphs.append(line.lower())
    sents = batched_sent_tokenize(paragraphs)
    my_write(save_sent_file, sents)


def collect_kw_occur_from_selected(files:list, keyword_dict:dict):
    for file in tqdm.tqdm(files):
        file_note = file[len(save_path)+1:].replace('/wiki_', ':')[:-4]
        with open(file) as f_in:
            for i, line in enumerate(csv.reader(f_in, delimiter='\t')):
                if i == 0:
                    continue
                head, tail = line[1], line[4]
                if head in keyword_dict:
                    keyword_dict[head].add('%s:%d' % (file_note, i))
                if tail in keyword_dict:
                    keyword_dict[tail].add('%s:%d' % (file_note, i))


def build_graph(save_selected_file_list:list, keyword_set):
    g = nx.Graph()
    print('Reading Co-occurrence lines')
    for f in tqdm.tqdm(save_selected_file_list):
        with open(f) as f_in:
            csv_reader = csv.reader(f_in, delimiter='\t')
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                if line[1] in keyword_set and line[4] in keyword_set:
                    g.add_edge(line[1], line[4])
    return g


if __name__ == '__main__':
    # Generate the save dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sub_folders = [sub for sub in os.listdir(wikipedia_dir)]
    save_sub_folders = [os.path.join(save_path, sub) for sub in sub_folders]
    wiki_sub_folders = [os.path.join(wikipedia_dir, sub) for sub in sub_folders]

    wiki_files = []
    save_sent_files = []
    save_selected_files = []
    for save_dir in save_sub_folders:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    for i in range(len(wiki_sub_folders)):
        files = [f for f in os.listdir(wiki_sub_folders[i])]
        wiki_files += [os.path.join(wiki_sub_folders[i], f) for f in files]
        save_sent_files += [os.path.join(save_sub_folders[i], f+'.dat') for f in files]
        save_selected_files += [os.path.join(save_sub_folders[i], f+'.tsv') for f in files]

    p = MyMultiProcessing(20)
    if sys.argv[1] == 'collect_sents':
        if len(sys.argv) == 2:
            sf = SentenceFilter(wikipedia_wordtree_file, wikipedia_token_file)
            def collect_sents(save_sent_file:str, save_selected_file:str):
                sents = my_read(save_sent_file)
                df = sf.list_operation(sents, use_id=True, keyword_only=True)
                df.to_csv(save_selected_file, sep='\t', index=False)
            input_list = [(save_sent_files[i], save_selected_files[i]) for i in range(len(save_sent_files))]
            p.run(collect_sents, input_list=input_list)
        
        elif len(sys.argv) == 6:
            file_name, output_file, use_id, keyword_only = sys.argv[2:6]
            sents = my_read(file_name)
            use_id = use_id == 'T'
            keyword_only = keyword_only == 'T'
            if keyword_only:
                sf = SentenceFilter(wikipedia_wordtree_file, wikipedia_token_file)
            else:
                sf = SentenceFilter()
            sf.list_operation(sents=sents, use_id=use_id, keyword_only=keyword_only).to_csv(output_file, sep='\t', index=False)

    
    elif sys.argv[1] == 'collect_kw_occur_from_selected':
        keyword_occur = {kw : set() for kw in my_read(wikipedia_keyword_file)}
        collect_kw_occur_from_selected(save_selected_files, keyword_occur)
        with open(keyword_occur_file, 'wb') as f_out:
            pickle.dump(keyword_occur, f_out)

    elif sys.argv[1] == 'build_graph':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        with open(keyword_occur_file, 'rb') as f_in:
            keyword_occur = pickle.load(f_in)
        keyword_connection_graph = build_graph(save_selected_files, keyword_occur)
        with open(keyword_connection_graph_file, 'wb') as f_out:
            pickle.dump(keyword_connection_graph, f_out)