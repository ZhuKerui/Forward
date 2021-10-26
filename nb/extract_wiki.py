# python extract_wiki.py collect_sents
# python extract_wiki.py collect_sents sentence_file output_file use_id[T/F] keyword_only[T/F]
# python extract_wiki.py collect_kw_occur_from_selected
# python extract_wiki.py collect_kw_occur_from_sents
# python extract_wiki.py build_graph_from_selected
# python extract_wiki.py build_graph_from_cooccur
# python extract_wiki.py build_graph_from_cooccur_v2
# python extract_wiki.py collect_cs_pages

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
sys.path.append('..')

from tools.BasicUtils import MyMultiProcessing, my_read, my_write, calculate_time
from tools.TextProcessing import normalize_text, remove_brackets, batched_sent_tokenize, nlp, find_span, sent_lemmatize, find_noun_phrases, find_dependency_path_from_tree, exact_match
from tools.DocProcessing import CoOccurrence


# Some constants
wikipedia_dir = '../../data/wikipedia/full_text-2021-03-20'
wikipedia_entity_file = 'data/extract_wiki/wikipedia_entity.tsv'
wikipedia_entity_norm_file = 'data/extract_wiki/wikipedia_entity_norm.tsv'
wikipedia_keyword_file = 'data/extract_wiki/keywords.txt'
wikipedia_keyword_filtered_file = 'data/extract_wiki/keywords_f.txt'
wikipedia_wordtree_file = 'data/extract_wiki/wordtree.pickle'
wikipedia_token_file = 'data/extract_wiki/tokens.txt'
save_path = 'data/extract_wiki/wiki_sent_collect'
keyword_occur_file = 'data/extract_wiki/keyword_occur.pickle'
keyword_connection_graph_file = 'data/extract_wiki/keyword_graph.pickle'
keyword_npmi_graph_file = 'data/extract_wiki/keyword_npmi_graph.pickle'
keyword_npmi_graph_file_v2 = 'data/extract_wiki/keyword_npmi_graph_v2.pickle'
keyword_count_file = 'data/extract_wiki/keyword_count.pickle'

w2vec_dump_file = 'data/extract_wiki/enwiki_20180420_win10_100d.pkl.bz2'
w2vec_keyword_file = 'data/extract_wiki/w2vec_keywords.txt'
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


@calculate_time
def build_graph_from_cooccur(keyword_file:str, files:list):
    g = nx.Graph(c=0)
    kw2idx = {kw:i for i, kw in enumerate(my_read(keyword_file))}
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
def build_graph_from_cooccur_v2(keyword_file:str, files:list):
    kw2idx = {kw:i for i, kw in enumerate(my_read(keyword_file))}
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


def build_graph_from_selected(save_selected_file_list:list, keyword_set):
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


def filter_keyword_by_freq(save_cooccur_file_list:list):
    c = Counter()
    for file in tqdm.tqdm(save_cooccur_file_list):
        with open(file) as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    c.update(line.split('\t'))
    return c


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


    elif sys.argv[1] == 'collect_kw_occur_from_sents':
        co = CoOccurrence(wikipedia_wordtree_file, wikipedia_token_file)
        def collect_kw_occur_from_sents(save_sent_file:str, save_cooccur_file:str):
            sents = my_read(save_sent_file)
            cooccur_list = ['\t'.join(co.line_operation(sent_lemmatize(sent))) for sent in sents]
            my_write(save_cooccur_file, cooccur_list)
        input_list = [(save_sent_files[i], save_cooccur_files[i]) for i in range(len(save_sent_files))]
        p.run(collect_kw_occur_from_sents, input_list=input_list)


    elif sys.argv[1] == 'build_graph_from_selected':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        with open(keyword_occur_file, 'rb') as f_in:
            keyword_occur = pickle.load(f_in)
        keyword_connection_graph = build_graph_from_selected(save_selected_files, keyword_occur)
        with open(keyword_connection_graph_file, 'wb') as f_out:
            pickle.dump(keyword_connection_graph, f_out)

    elif sys.argv[1] == 'build_graph_from_cooccur':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        keyword_npmi_graph = build_graph_from_cooccur(wikipedia_keyword_filtered_file, save_cooccur_files)
        with open(keyword_npmi_graph_file, 'wb') as f_out:
            pickle.dump(keyword_npmi_graph, f_out)

    elif sys.argv[1] == 'build_graph_from_cooccur_v2':
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        keyword_npmi_graph = build_graph_from_cooccur_v2(wikipedia_keyword_filtered_file, save_cooccur_files)
        with open(keyword_npmi_graph_file_v2, 'wb') as f_out:
            pickle.dump(keyword_npmi_graph, f_out)

    elif sys.argv[1] == 'collect_cs_pages':
        input_list = [(wiki_files[i], cs_keyword_file, save_cs_sent_files[i]) for i in range(len(wiki_files))]
        _ = p.run(get_sentence_from_page_using_keyword, input_list)
