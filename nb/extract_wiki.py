# python extract_wiki.py collect_sents
# python extract_wiki.py collect_sents sentence_file output_file use_id[T/F] keyword_only[T/F]
# python extract_wiki.py collect_ent_occur_from_selected
# python extract_wiki.py collect_kw_occur_from_sents
# python extract_wiki.py build_graph_from_selected
# python extract_wiki.py build_graph_from_cooccur
# python extract_wiki.py build_graph_from_cooccur_v2
# python extract_wiki.py collect_cs_pages
# python extract_wiki.py collect_dataset
# python extract_wiki.py generate_graph

import re
import os
import tqdm
import csv
import math
import pickle
import networkx as nx
import pandas as pd
from urllib.parse import unquote
from typing import List
from collections import Counter
import sys
import linecache
import numpy as np
sys.path.append('..')

from tools.BasicUtils import MyMultiProcessing, my_read, my_write, calculate_time
from tools.TextProcessing import normalize_text, remove_brackets, batched_sent_tokenize, nlp, find_span, sent_lemmatize, find_noun_phrases, find_dependency_path_from_tree, exact_match
from tools.DocProcessing import CoOccurrence


# Some constants
wikipedia_dir = '../../data/wikipedia/full_text-2021-03-20'
wikipedia_entity_file = 'data/extract_wiki/wikipedia_entity.tsv'
wikipedia_entity_norm_file = 'data/extract_wiki/wikipedia_entity_norm.tsv'
wikipedia_keyword_file = 'data/extract_wiki/keywords.txt'
wikipedia_wordtree_file = 'data/extract_wiki/wordtree.pickle'
wikipedia_token_file = 'data/extract_wiki/tokens.txt'
save_path = 'data/extract_wiki/wiki_sent_collect'
entity_occur_file = 'data/extract_wiki/entity_occur.pickle'
keyword_connection_graph_file = 'data/extract_wiki/keyword_graph.pickle' # deprecated
keyword_npmi_graph_file = 'data/extract_wiki/keyword_npmi_graph.pickle'  # deprecated
keyword_npmi_graph_file_v2 = 'data/extract_wiki/keyword_npmi_graph_v2.pickle' # deprecated
graph_file = 'data/extract_wiki/graph.pickle'

w2vec_dump_file = 'data/extract_wiki/enwiki_20180420_win10_100d.pkl.bz2'
w2vec_keyword_file = 'data/extract_wiki/w2vec_keywords.txt'
w2vec_entity_file = 'data/extract_wiki/w2vec_entity.txt'
w2vec_wordtree_file = 'data/extract_wiki/w2vec_wordtree.pickle'
w2vec_token_file = 'data/extract_wiki/w2vec_tokens.txt'
w2vec_keyword2idx_file = 'data/extract_wiki/w2vec_keyword2idx.pickle'

cs_keyword_file = 'data/extract_wiki/cs_keyword.txt'
cs_raw_keyword_file = 'data/extract_wiki/cs_raw_keyword.txt'
cs_wordtree_file = 'data/extract_wiki/cs_wordtree.pickle'
cs_token_file = 'data/extract_wiki/cs_token.txt'
cs_pair_file = 'data/extract_wiki/cs_pair.gpickle'

test_path = 'data/extract_wiki/wiki_sent_test'
path_test_file = 'data/extract_wiki/wiki_sent_test/path_test.tsv'
cs_path_test_file = 'data/extract_wiki/wiki_sent_test/cs_path_test.tsv'

path_pattern_count_file = 'data/extract_wiki/path_pattern.pickle'

patterns = [
            r'i_nsubj attr( prep pobj)*( compound){0, 1}', 
            r'i_nsubj( conj)* dobj( acl prep pobj( conj)*){0,1}', 
            r'i_nsubj( prep pobj)+( conj)*', 
            r'i_nsubj advcl dobj( acl attr){0,1}', 
            r'appos( conj)*', 
            r'appos acl prep pobj( conj)*', 
            r'i_nsubjpass( conj)*( prep pobj)+( conj)*', 
            r'i_nsubjpass prep pobj acl dobj', 
            r'i_dobj prep pobj( conj)*'
            # 'acl prep pobj( conj)*'
        ]
        
# Some task specific classes
class SentenceFilter:
    def __init__(self, wordtree_file:str=None, token_file:str=None):
        if wordtree_file and token_file:
            self.co_occur_finder = CoOccurrence(wordtree_file, token_file)
        else:
            self.co_occur_finder = None
        
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
    sents = [normalize_text(sent) for sent in sents]
    my_write(save_sent_file, sents)


def collect_ent_occur_from_selected(files:list, keyword_dict:dict):
    for file in tqdm.tqdm(files):
        with open(file) as f_in:
            for i, line in enumerate(csv.reader(f_in, delimiter='\t')):
                if i == 0:
                    continue
                head, tail, idx = line[3], line[6], line[7]
                idx = '%s:%d' % (idx.rsplit(':', 1)[0], i)
                keyword_dict[head].add(idx)
                keyword_dict[tail].add(idx)


@calculate_time
def build_graph_from_cooccur(entity_file:str, files:list):
    g = nx.Graph(c=0)
    kw2idx = {kw:i for i, kw in enumerate(my_read(entity_file))}
    g.add_nodes_from(range(len(kw2idx)), c=0)
    print('Reading Co-occurrence lines')
    for file in tqdm.tqdm(files):
        with open(file) as f_in:
            for line in f_in:
                kws = [kw2idx.get(kw) for kw in line.strip().split('\t')]
                kws = [kw for kw in kws if kw]
                kw_num = len(kws)
                if kw_num < 2:
                    continue
                g.graph['c'] += kw_num * (kw_num - 1) / 2
                for i in range(kw_num):
                    g.nodes[kws[i]]['c'] += (kw_num - 1)
                    for j in range(i+1, kw_num):
                        if not g.has_edge(kws[i], kws[j]):
                            g.add_edge(kws[i], kws[j], c=1)
                        else:
                            g.edges[kws[i], kws[j]]['c'] += 1
    print('Reading Done! NPMI analysis starts...')
    Z = float(g.graph['c'])
    for e, attr in g.edges.items():
        h_x = math.log2(g.nodes[e[0]]['c'] / Z)
        h_y = math.log2(g.nodes[e[1]]['c'] / Z)
        h_xy = math.log2(attr['c'] / Z)
        attr['npmi'] = (h_x + h_y) / h_xy - 1
    print('NPMI analysis Done')
    return g


@calculate_time
def build_graph_from_cooccur_v2(entity_file:str, files:list):
    kw2idx = {kw:i for i, kw in enumerate(my_read(entity_file))}
    g = {idx : {'C':0} for idx in range(len(kw2idx))}
    g['C'] = 0
    print('Reading Co-occurrence lines')
    for file in tqdm.tqdm(files):
        with open(file) as f_in:
            for line in f_in:
                kws = [kw2idx.get(kw) for kw in line.strip().split('\t')]
                kws = [kw for kw in kws if kw is not None]
                kw_num = len(kws)
                if kw_num < 2:
                    continue
                g['C'] += kw_num * (kw_num - 1) / 2
                for i in range(kw_num):
                    g[kws[i]]['C'] += (kw_num - 1)
                    for j in range(i+1, kw_num):
                        kw1, kw2 = (kws[i], kws[j]) if kws[i] < kws[j] else (kws[j], kws[i])
                        if kw2 not in g[kw1]:
                            g[kw1][kw2] = {'C':1}
                        else:
                            g[kw1][kw2]['C'] += 1
    print('Reading Done! NPMI analysis starts...')
    Z = float(g['C'])
    for kw1 in range(len(kw2idx)):
        if len(g[kw1]) <= 1:
            continue
        h_1 = math.log2(g[kw1]['C'] / Z)
        for kw2, val in g[kw1].items():
            if kw2 == 'C':
                continue
            h_2 = math.log2(g[kw2]['C'] / Z)
            h_12 = math.log2(val['C'] / Z)
            val['NPMI'] = (h_1 + h_2) / h_12 - 1
    print('NPMI analysis Done')
    return g


def build_graph_from_selected(save_selected_file_list:list):
    g = nx.Graph()
    print('Reading Co-occurrence lines')
    for f in tqdm.tqdm(save_selected_file_list):
        with open(f) as f_in:
            csv_reader = csv.reader(f_in, delimiter='\t')
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                ent1, ent2 = line[3], line[6]
                if not g.has_node(ent1):
                    g.add_node(ent1, c=1)
                else:
                    g.nodes[ent1]['c'] += 1

                if not g.has_node(ent2):
                    g.add_node(ent2, c=1)
                else:
                    g.nodes[ent2]['c'] += 1
                    
                if not g.has_edge(ent1, ent2):
                    g.add_edge(ent1, ent2, c=1)
                else:
                    g.edges[ent1, ent2]['c'] += 1
    return g


def generate_graph(files:list):
    g = nx.Graph()
    for file in tqdm.tqdm(files):
        with open(file) as f_in:
            for i, record in enumerate(f_in):
                if i == 0:
                    continue
                record = record.strip().split('\t')
                if not g.has_edge(record[3], record[6]):
                    g.add_edge(record[3], record[6], score=float(record[-1]), sent=record[7])
                else:
                    data = g.get_edge_data(record[3], record[6])
                    new_score = float(record[-1])
                    if data['score'] < new_score:
                        data['score'] = new_score
                        data['sent'] = record[7]
    return g


def line2note(filename:str, line_idx:int, posfix='.dat'):
    posfix_len = len(posfix)
    return filename[len(save_path)+1:].replace('/wiki_', ':')[:-posfix_len] + ':' + str(line_idx)


def note2line(note:str, posfix='.dat'):
    sub_folder, sub_file, line_idx = note.split(':')
    return linecache.getline(save_path + '/' + sub_folder + '/wiki_' + sub_file + posfix, int(line_idx)+1)


def get_sentence_from_page_using_keyword(wiki_file:str, keyword_file:str, save_sent_file):
    '''
    Collect wikipedia pages from a wiki file whose title (brackets removed) is in the keyword file
    '''
    paragraphs = []
    keywords = set(my_read(keyword_file))
    with open(wiki_file) as f_in:
        # Remove useless lines
        entity_name = ''
        useful_page = False
        for line in f_in:
            line = line.strip()
            if not line or line == entity_name:
                continue
            if re.match(r'^(<doc id=")', line):
                entity_name = line[re.search(r'title="', line).end():re.search(r'">', line).start()]
                # entity_id = line[9:re.search(r'" url="', line).start()]
                keyword = remove_brackets(normalize_text(entity_name))
                useful_page = keyword in keywords
            elif useful_page:
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
    sents = [normalize_text(sent) for sent in sents]
    my_write(save_sent_file, sents)


# Basic collection, contain entities similarity, keyword name, keyword span, corresponding entity, sentence note and dep path
basic_columns=['sim', 'kw1', 'kw1_span', 'kw1_ent', 'kw2', 'kw2_span', 'kw2_ent', 'sent', 'path']

def basic_process(doc, pairs):
    data = []
    for item in pairs:
        kw1_spans = find_span(doc, item['kw1'], True)
        kw2_spans = find_span(doc, item['kw2'], True)
        for kw1_span in kw1_spans:
            for kw2_span in kw2_spans:
                path = find_dependency_path_from_tree(doc, kw1_span, kw2_span)
                if not path:
                    continue
                item['kw1_span'] = (kw1_span[0].i, kw1_span[-1].i)
                item['kw2_span'] = (kw2_span[0].i, kw2_span[-1].i)
                item['path'] = path
                data.append(item.copy())
    return data

# Feature collection, contains subject/object full span and keyword-entity recall
def get_back(doc, idx):
    while doc[idx].dep_ == 'compound' or doc[idx].dep_ == 'amod':
        idx = doc[idx].head.i
    return idx


def get_ahead(doc, idx):
    mod_exist = True
    while mod_exist:
        c_list = [c for c in doc[idx].children]
        c_dep_list = [c.dep_ for c in c_list]
        mod_exist = False
        for i, dep in enumerate(c_dep_list):
            if 'amod' == dep or 'compound' == dep:
                idx_ = c_list[i].i
                if idx_ < idx:
                    idx = idx_
                    mod_exist = True
    return idx

feature_columns=['sim', 'kw1', 'kw1_span', 'kw1_ent', 'kw2', 'kw2_span', 'kw2_ent', 'sent', 'path', 'kw1_full_span', 'kw1_recall', 'kw2_full_span', 'kw2_recall']

def feature_process(doc, pairs):
    data = []
    for item in pairs:
        kw1_spans = find_span(doc, item['kw1'], True)
        kw2_spans = find_span(doc, item['kw2'], True)
        kw1_clean = ' '.join(re.sub(r'[^a-z0-9\s]', '', item['kw1']).split())
        kw2_clean = ' '.join(re.sub(r'[^a-z0-9\s]', '', item['kw2']).split())
        
        for kw1_span in kw1_spans:
            for kw2_span in kw2_spans:
                path = find_dependency_path_from_tree(doc, kw1_span, kw2_span)
                if not path:
                    continue
                item['kw1_span'] = (kw1_span[0].i, kw1_span[-1].i)
                item['kw2_span'] = (kw2_span[0].i, kw2_span[-1].i)
                item['path'] = path
                # Calculate subject and object coverage
                kw1_right_most = get_back(doc, kw1_span[-1].i)
                kw1_left_most = min(get_ahead(doc, kw1_right_most), get_ahead(doc, kw1_span[0].i), kw1_span[0].i)
                
                const_num = 0.5
                
                item['kw1_full_span'] = ' '.join(re.sub(r'[^a-z0-9\s]', '', doc[kw1_left_most : kw1_right_most+1].text).split())
                item['kw1_recall'] = np.log(kw1_clean.count(' ') + 1 + const_num) / np.log(item['kw1_full_span'].count(' ') + 1 + const_num)

                kw2_right_most = get_back(doc, kw2_span[-1].i)
                kw2_left_most = min(get_ahead(doc, kw2_right_most), get_ahead(doc, kw2_span[0].i), kw2_span[0].i)
                
                item['kw2_full_span'] = ' '.join(re.sub(r'[^a-z0-9\s]', '', doc[kw2_left_most : kw2_right_most+1].text).split())
                item['kw2_recall'] = np.log(kw2_clean.count(' ') + 1 + const_num) / np.log(item['kw2_full_span'].count(' ') + 1 + const_num)

                data.append(item.copy())
    return data


def reverse_path(path:str):
    path = path.split()
    r_path = ' '.join(['i_' + token if token[:2] != 'i_' else token[2:] for token in reversed(path)])
    return r_path

def gen_pattern(path:str):
    if 'i_nsubj' not in path:
        path = reverse_path(path)
    path = ' '.join([token for token in path.split() if 'appos' not in token and 'conj' not in token])
    return path

def cal_coverage(sent:str, kw1:str, kw2:str, path:str):
    return (len(kw1.split()) + len(kw2.split()) + len(path.split()) - 1) / len(normalize_text(sent).split())


if __name__ == '__main__':
    # Generate the save dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sub_folders = [sub for sub in os.listdir(wikipedia_dir)]
    save_sub_folders = [os.path.join(save_path, sub) for sub in sub_folders]
    wiki_sub_folders = [os.path.join(wikipedia_dir, sub) for sub in sub_folders]

    wiki_files = []
    save_sent_files = []
    save_cooccur_files = []
    save_cs_sent_files = []
    save_selected_files = []

    for save_dir in save_sub_folders:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    for i in range(len(wiki_sub_folders)):
        files = [f for f in os.listdir(wiki_sub_folders[i])]
        wiki_files += [os.path.join(wiki_sub_folders[i], f) for f in files]
        save_sent_files += [os.path.join(save_sub_folders[i], f+'.dat') for f in files]
        save_cooccur_files += [os.path.join(save_sub_folders[i], f+'_co.dat') for f in files]
        save_cs_sent_files += [os.path.join(save_sub_folders[i], f+'_cs.dat') for f in files]
        save_selected_files += [os.path.join(save_sub_folders[i], f+'_se.dat') for f in files]

    p = MyMultiProcessing(20)
    if sys.argv[1] == 'collect_sents':
        if len(sys.argv) == 2:
            # sf = SentenceFilter(wikipedia_wordtree_file, wikipedia_token_file)
            # def collect_sents(save_sent_file:str, save_selected_file:str):
            #     sents = my_read(save_sent_file)
            #     df = sf.list_operation(sents, use_id=True, keyword_only=True)
            #     df.to_csv(save_selected_file, sep='\t', index=False)
            # input_list = [(save_sent_files[i], save_selected_files[i]) for i in range(len(save_sent_files))]
            # p.run(collect_sents, input_list=input_list)
            pass
        
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

    
    elif sys.argv[1] == 'collect_ent_occur_from_selected':
        keyword_occur = {kw : set() for kw in my_read(w2vec_entity_file)}
        collect_ent_occur_from_selected(save_selected_files, keyword_occur)
        with open(entity_occur_file, 'wb') as f_out:
            pickle.dump(keyword_occur, f_out)


    elif sys.argv[1] == 'collect_kw_occur_from_sents':
        co = CoOccurrence(w2vec_wordtree_file, w2vec_token_file)
        def collect_kw_occur_from_sents(save_sent_file:str, save_cooccur_file:str):
            sents = my_read(save_sent_file)
            cooccur_list = ['\t'.join(co.line_operation(sent_lemmatize(sent))) for sent in sents]
            my_write(save_cooccur_file, cooccur_list)
        input_list = [(save_sent_files[i], save_cooccur_files[i]) for i in range(len(save_sent_files))]
        p.run(collect_kw_occur_from_sents, input_list=input_list)


    elif sys.argv[1] == 'build_graph_from_selected':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        keyword_connection_graph = build_graph_from_selected(save_selected_files)
        with open(keyword_connection_graph_file, 'wb') as f_out:
            pickle.dump(keyword_connection_graph, f_out)
            
    elif sys.argv[1] == 'generate_graph':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        graph = generate_graph(save_selected_files)
        with open(graph_file, 'wb') as f_out:
            pickle.dump(graph, f_out)

    # elif sys.argv[1] == 'build_graph_from_cooccur':
    #     # Load keyword occur dict which has occurance record for all keywords in selected sentences
    #     keyword_npmi_graph = build_graph_from_cooccur(wikipedia_keyword_filtered_file, save_cooccur_files)
    #     with open(keyword_npmi_graph_file, 'wb') as f_out:
    #         pickle.dump(keyword_npmi_graph, f_out)

    # elif sys.argv[1] == 'build_graph_from_cooccur_v2':
    #     # Load keyword occur dict which has occurance record for all keywords in selected sentences
    #     keyword_npmi_graph = build_graph_from_cooccur_v2(wikipedia_keyword_filtered_file, save_cooccur_files)
    #     with open(keyword_npmi_graph_file_v2, 'wb') as f_out:
    #         pickle.dump(keyword_npmi_graph, f_out)

    elif sys.argv[1] == 'collect_cs_pages':
        input_list = [(wiki_files[i], cs_keyword_file, save_cs_sent_files[i]) for i in range(len(wiki_files))]
        _ = p.run(get_sentence_from_page_using_keyword, input_list)

    elif sys.argv[1] == 'collect_dataset':
        import bz2
        from wikipedia2vec import Wikipedia2Vec
        from sklearn.metrics.pairwise import cosine_similarity

        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)

        with open(w2vec_keyword2idx_file, 'rb') as f_in:
            my_mention_dict = pickle.load(f_in)

        co = CoOccurrence(w2vec_wordtree_file, w2vec_token_file)
        
        with open(path_pattern_count_file, 'rb') as f_in:
            c = pickle.load(f_in)

        max_cnt = c.most_common(1)[0][1]
        log_max_cnt = np.log(max_cnt+1)

        def cal_freq(path:str):
            cnt = c.get(path)
            cnt = (cnt if cnt else 0.5) + 1
            return np.log(cnt) / log_max_cnt

        def collect_paths_in_bg(test_file:str, process_func, columns:list, posfix:str='.dat'):
            # Build test data
            with open(test_file) as f_in:
                data = []
                for line_idx, line in enumerate(f_in.readlines()):
                    sent_note = line2note(test_file, line_idx, posfix=posfix)
                    line = line.strip()
                    co_kws = list(co.line_operation(sent_lemmatize(line)))
                    if len(co_kws) < 2:
                        continue
                    certain_ent_list = []
                    certain_ent_kw_list = []
                    uncertain_ent_list = []
                    uncertain_ent_kw_list = []
                    for kw in co_kws:
                        idxs = my_mention_dict[kw]
                        if len(idxs) == 1:
                            certain_ent_kw_list.append(kw)
                            certain_ent_list.append(w2vec.dictionary.get_entity_by_index(idxs[0]))
                        else:
                            uncertain_ent_kw_list.append(kw)
                            uncertain_ent_list.append([w2vec.dictionary.get_entity_by_index(idx) for idx in idxs])
                    
                    certain_ent_matrix = np.array([w2vec.get_vector(ent) for ent in certain_ent_list])
                    uncertain_ent_matrix_list = [np.array([w2vec.get_vector(ent) for ent in ent_list]) for ent_list in uncertain_ent_list]
                    pairs = []
                    certain_len = len(certain_ent_list)
                    uncertain_len = len(uncertain_ent_list)
                    if certain_len >= 1:
                        # Collect pairs between certain entities
                        result = cosine_similarity(certain_ent_matrix, certain_ent_matrix) - np.identity(certain_len)
                        for i in range(certain_len):
                            for j in range(i+1, certain_len):
                                pairs.append({'kw1':certain_ent_kw_list[i], 'kw2':certain_ent_kw_list[j], 'sim':float(result[i, j]), 'sent':sent_note, 
                                    'kw1_ent':certain_ent_list[i].title, 
                                    'kw2_ent':certain_ent_list[j].title})
                        # Collect pairs between certain and uncertain entities
                        for i in range(uncertain_len):
                            result = cosine_similarity(certain_ent_matrix, uncertain_ent_matrix_list[i])
                            for j in range(certain_len):
                                idx = np.argmax(result[j])
                                pairs.append({'kw1':uncertain_ent_kw_list[i], 'kw2':certain_ent_kw_list[j], 'sim':float(result[j, idx]), 'sent':sent_note, 
                                    'kw1_ent':uncertain_ent_list[i][idx].title, 
                                    'kw2_ent':certain_ent_list[j].title})
                    if uncertain_len >= 2:
                        # Collect pairs between uncertain entities
                        for i in range(uncertain_len):
                            for j in range(i+1, uncertain_len):
                                result = cosine_similarity(uncertain_ent_matrix_list[i], uncertain_ent_matrix_list[j])
                                idx = np.argmax(result)
                                row = int(idx / result.shape[1])
                                col = idx % result.shape[1]
                                pairs.append({'kw1':uncertain_ent_kw_list[i], 'kw2':uncertain_ent_kw_list[j], 'sim':float(result[row, col]), 'sent':sent_note, 
                                    'kw1_ent':uncertain_ent_list[i][row].title, 
                                    'kw2_ent':uncertain_ent_list[j][col].title})
                    pairs = [item for item in pairs if item['sim'] > 0.5]
                    if pairs:
                        doc = nlp(line)
                        for item in process_func(doc, pairs):
                            item['pattern'] = gen_pattern(item['path'])
                            item['pattern_freq'] = cal_freq(item['pattern'])
                            item['coverage'] = cal_coverage(line, item['kw1'], item['kw2'], item['path'])
                            item['score'] = (item['pattern_freq'] * (item['kw1_recall'] + item['kw2_recall']) / 2 * item['coverage'])**0.33
                            if item['score'] > 0.5:
                                data.append(item)
                pd.DataFrame(data=data, columns=columns).to_csv(test_file[:-len(posfix)]+'_se'+posfix, sep='\t', index=False)

        input_list = [(file, feature_process, feature_columns + ['pattern', 'pattern_freq', 'coverage', 'score']) for file in save_sent_files]
        _ = p.run(collect_paths_in_bg, input_list)