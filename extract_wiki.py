# python extract_wiki.py collect_sent_and_cooccur
# python extract_wiki.py collect_ent_occur_from_cooccur
# python extract_wiki.py collect_pattern_freq
# python extract_wiki.py collect_dataset
# python extract_wiki.py collect_score_function_eval_dataset
# python extract_wiki.py generate_graph
# python extract_wiki.py generate_single_sent_graph
# python extract_wiki.py collect_one_hop_sample_from_single_sent_graph
# python extract_wiki.py collect_second_level_sample

from copy import deepcopy
import re
import os
import time
from collections import Counter
from queue import SimpleQueue
import json
import random
random.seed(0)
from nltk.tokenize import sent_tokenize
from scipy import rand
import tqdm
import csv
import networkx as nx
import pandas as pd
from urllib.parse import unquote
import sys
import linecache
import numpy as np
from typing import List, Tuple
import bz2
import itertools
from wikipedia2vec import Wikipedia2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
sys.path.append('..')

from tools.BasicUtils import MyMultiProcessing, my_read, my_write, my_read_pickle, my_write_pickle
from tools.TextProcessing import (remove_brackets, find_root_in_span, spacy, 
                                  nlp, find_span, sent_lemmatize, find_dependency_path_from_tree,
                                  build_word_tree_v2)
from tools.DocProcessing import CoOccurrence


# Some constants
save_path = 'data/extract_wiki/wiki_sent_collect'
test_path = 'data/extract_wiki/wiki_sent_test'
wikipedia_dir = '../data/wikipedia/full_text-2021-03-20'
sub_folders = [sub for sub in os.listdir(wikipedia_dir)]
save_sub_folders = [os.path.join(save_path, sub) for sub in sub_folders]
wiki_sub_folders = [os.path.join(wikipedia_dir, sub) for sub in sub_folders]

wiki_files = []
save_sent_files = []
save_cooccur_files = []
save_cooccur__files = []
save_pair_files = []
save_selected_files = []
save_title_files = []

for i in range(len(wiki_sub_folders)):
    files = [f for f in os.listdir(wiki_sub_folders[i])]
    wiki_files += [os.path.join(wiki_sub_folders[i], f) for f in files]
    save_sent_files += [os.path.join(save_sub_folders[i], f+'.dat') for f in files]
    save_cooccur_files += [os.path.join(save_sub_folders[i], f+'_co.dat') for f in files]
    save_cooccur__files += [os.path.join(save_sub_folders[i], f+'_co_.dat') for f in files]
    save_pair_files += [os.path.join(save_sub_folders[i], f+'_pr.dat') for f in files]
    save_selected_files += [os.path.join(save_sub_folders[i], f+'_se.dat') for f in files]
    save_title_files += [os.path.join(save_sub_folders[i], f+'_ti.dat') for f in files]

wikipedia_entity_file = 'data/extract_wiki/wikipedia_entity.tsv'

w2vec_dump_file = 'data/extract_wiki/enwiki_20180420_win10_100d.pkl.bz2'
w2vec_keyword2idx_file = 'data/extract_wiki/w2vec_keyword2idx.pickle'

entity_occur_from_cooccur_file = 'data/extract_wiki/entity_occur_from_cooccur.pickle'
path_test_file = 'data/extract_wiki/wiki_sent_test/path_test.tsv'
path_pattern_count_file = 'data/extract_wiki/path_pattern.pickle'
sub_path_pattern_count_file = 'data/extract_wiki/sub_path_pattern.pickle'

graph_file = 'data/extract_wiki/graph.pickle'
single_sent_graph_file = 'data/extract_wiki/single_sent_graph.pickle'

p = MyMultiProcessing(10)

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
        
# Some private variables
const_num = 0.5
kw1_str = 'kw1'
kw2_str = 'kw2'
kw1_ent_str = 'kw1_ent'
kw2_ent_str = 'kw2_ent'
sent_str = 'sent'
kw1_span_str = 'kw1_span'
kw2_span_str = 'kw2_span'
dep_path_str = 'dep_path'
dep_coverage_str = 'dep_coverage'
sim_str = 'sim'
pattern_str = 'pattern'
pattern_freq_str = 'pattern_freq'
score_str = 'score'
similar_threshold = 0.5
max_sentence_length = 50
min_sentence_length = 5

# Some task specific classes


# Some helper functions
def collect_wiki_entity(file:str):
    return ['%s\t%s' % (line[9:re.search(r'" url="', line).start()], line[re.search(r'title="', line).end():re.search(r'">', line).start()]) for line in open(file).readlines() if re.match(r'^(<doc id=")', line) and line.isascii()]


def gen_kw_from_wiki_ent(wiki_ent:str, lower:bool=True):
    if lower:
        wiki_ent = wiki_ent.lower()
    bracket_removed = remove_brackets(wiki_ent).strip()
    if bracket_removed:
        return ' '.join(word_tokenize(bracket_removed.split(',')[0]))
    else:
        return ' '.join(word_tokenize(wiki_ent))


def get_sentence(wiki_file:str, save_sent_file:str, save_cooccur_file:str, save_title_file:str):
    sents = []
    cooccurs = []
    page_titles = []
    with open(wiki_file) as f_in:
        # Collect pages
        page_name = ''
        page_kw = ''
        wordtree, token2idx, kw2ent_map = {}, {}, {}
        for line in f_in:
            line = line.strip()
            if not line or line == page_name or line == '</doc>':
                continue
            if re.match(r'^(<doc id=")', line):
                page_name = ' '.join(line[re.search(r'title="', line).end():re.search(r'">', line).start()].split())
                page_kw = gen_kw_from_wiki_ent(page_name, True)
                wordtree, token2idx = build_word_tree_v2([page_kw])
                kw2ent_map = {page_kw : page_name}
            else:
                kw2ent_map[page_kw] = page_name
                links = re.findall(r'<a href="[^"]*">[^<]*</a>', line)
                new_kws = []
                for l in links:
                    breakpoint = l.index('">')
                    entity_name = ' '.join(unquote(l[9:breakpoint]).split())
                    kw = gen_kw_from_wiki_ent(entity_name, False)
                    kw_lower = kw.lower()
                    if kw == '':
                        print(wiki_file, line, entity_name)
                    else:
                        kw2ent_map[kw_lower] = entity_name
                        new_kws.append(kw_lower)
                    # Replace link with plain text
                    kw_in_text:str = l[breakpoint+2:-4]
                    kw_in_text_lower = kw_in_text.lower()
                    if kw_lower[-len(kw_in_text):] == kw_in_text_lower:
                        if kw_in_text.islower():
                            line = line.replace(l, kw_lower)
                        else:
                            line = line.replace(l, kw)
                    else:
                        line = line.replace(l, kw_in_text)
                paragraph = sent_tokenize(line)
                wordtree, token2idx = build_word_tree_v2(new_kws, old_MyTree=wordtree, old_token2idx=token2idx)
                co = CoOccurrence(wordtree, token2idx)
                for sent in paragraph:
                    sent = remove_brackets(sent)
                    reformed_sent = word_tokenize(sent)
                    reformed_sent = sent_lemmatize(sent)
                    reformed_sent = [text.lower() for text in reformed_sent]
                    kws = co.line_operation(reformed_sent)
                    sents.append(sent)
                    cooccurs.append('\t'.join([kw2ent_map[kw] for kw in kws]))
                page_titles += [page_name] * len(paragraph)
                
    my_write(save_sent_file, sents)
    my_write(save_cooccur_file, cooccurs)
    my_write(save_title_file, page_titles)
            
            
def line2note(filename:str, line_idx:int, posfix='.dat'):
    posfix_len = len(posfix)
    return filename[len(save_path)+1:].replace('/wiki_', ':')[:-posfix_len] + ':' + str(line_idx)


def note2line(note:str, posfix='.dat'):
    sub_folder, sub_file, line_idx = note.split(':')
    return linecache.getline(save_path + '/' + sub_folder + '/wiki_' + sub_file + posfix, int(line_idx)+1)


# Feature collection, contains subject/object full span and keyword-entity recall
def get_back(doc, idx):
    while doc[idx].dep_ == 'compound':
        idx = doc[idx].head.i
    return idx


def get_ahead(doc, idx):
    mod_exist = True
    while mod_exist:
        c_list = [c for c in doc[idx].children]
        c_dep_list = [c.dep_ for c in c_list]
        mod_exist = False
        for i, dep in enumerate(c_dep_list):
            if 'compound' == dep:
                idx_ = c_list[i].i
                if idx_ < idx:
                    idx = idx_
                    mod_exist = True
    return idx

# feature_columns=[__sim, __kw1, __kw1_span, __kw1_ent, __kw2, __kw2_span, __kw2_ent, __sent, __dep_path, __pattern, __kw1_full_span, __kw1_recall, __kw2_full_span, __kw2_recall, __dep_coverage, __surface_coverage]
feature_columns=[sim_str, kw1_str, kw1_span_str, kw1_ent_str, kw2_str, kw2_span_str, kw2_ent_str, sent_str, dep_path_str, pattern_str, dep_coverage_str]

def get_phrase_full_span(doc, phrase_span):
    phrase_right_most = get_back(doc, phrase_span[-1].i)
    phrase_left_most = min(get_ahead(doc, phrase_right_most), get_ahead(doc, phrase_span[0].i), phrase_span[0].i)
    return (phrase_left_most, phrase_right_most)

def cal_kw_recall(kw:str, full_phrase:str):
    return np.log(kw.count(' ') + 1 + const_num) / np.log(full_phrase.count(' ') + 1 + const_num)

def generate_clean_phrase(phrase:str):
    return ' '.join(re.sub(r'[^a-z0-9\s]', '', phrase).split())


def sentence_decompose(doc, kw1:str, kw2:str):
    kw1_spans = find_span(doc, kw1, True, True)
    kw2_spans = find_span(doc, kw2, True, True)
    data = []
    for kw1_span in kw1_spans:
        for kw2_span in kw2_spans:
            kw1_left_most, kw1_right_most = get_phrase_full_span(doc, kw1_span)
            kw2_left_most, kw2_right_most = get_phrase_full_span(doc, kw2_span)
            if kw1_left_most != kw1_span[0].i or kw1_right_most != kw1_span[-1].i or kw2_left_most != kw2_span[0].i or kw2_right_most != kw2_span[-1].i:
                # full span and keyword span don't match
                continue
            kw1_steps, kw2_steps, branch = find_dependency_info_from_tree(doc, kw1_span, kw2_span)
            if not branch.any():
                continue
            path = get_path(doc, kw1_steps, kw2_steps)
            pattern = gen_pattern(path)
            if not pattern.startswith('i_nsubj'):
                continue
            data.append((kw1_span, kw2_span, branch, path, pattern))
    return data


class FeatureProcess:
    def __init__(self, sub_path_pattern_file:str):
        c, log_max_cnt = load_pattern_freq(sub_path_pattern_file)
        self.c = c
        self.log_max_cnt = log_max_cnt
        
    
    def expand_dependency_info_from_tree(self, doc, path:np.ndarray):
        dep_path:list = (np.arange(*path.shape)[path!=0]).tolist()
        for element in dep_path:
            if doc[element].dep_ == 'conj':
                path[doc[element].head.i] = 0
        paths = collect_sub_dependency_path(doc, path)
        paths = [item for item in paths if item[1].split()[0] in modifier_dependencies]
        for p in paths:
            pattern = gen_sub_dep_path_pattern(p[1])
            if pattern == '':
                path[p[2]] = path[p[0]]
            else:
                path[p[2]] = cal_freq_from_path(pattern, self.c, self.log_max_cnt)
            
    def feature_process(self, doc, kw1:str, kw2:str)->List[dict]:
        data = []
        punct_mask = np.array([token.dep_ != 'punct' for token in doc])
        for kw1_span, kw2_span, branch, path, pattern in sentence_decompose(doc, kw1, kw2):
            self.expand_dependency_info_from_tree(doc, branch)
            data.append({kw1_span_str : (kw1_span[0].i, kw1_span[-1].i),
                         kw2_span_str : (kw2_span[0].i, kw2_span[-1].i),
                         pattern_str : pattern, 
                         dep_path_str : path, 
                         dep_coverage_str : branch[punct_mask].mean()
                        })
        return data
    
    def batched_feature_process(self, sent:str, pairs):
        data = []
        doc = nlp(sent)
        if len(doc) > max_sentence_length or len(doc) < min_sentence_length:
            return []
        for item in pairs:
            # Calculate calculate dependency coverage
            temp_data = self.feature_process(doc, item[kw1_str], item[kw2_str])
            for d in temp_data:
                d.update(item)
            data.extend(temp_data)
        return data


def gen_sub_dep_path_pattern(path:str):
    return ' '.join(path.replace('compound', '').replace('conj', '').replace('appos', '').split())

def collect_sub_dep_path(doc, kw1:str, kw2:str)->List[dict]:
    data = []
    for kw1_span, kw2_span, branch, path, pattern in sentence_decompose(doc, kw1, kw2):
        ans = [str(item[1]) for item in collect_sub_dependency_path(doc, branch)]
        ans = [gen_sub_dep_path_pattern(item) for item in ans if 'punct' not in item]
        ans = [item for item in ans if item]
        data.extend(ans)
    return data


def batched_collect_sub_dep_path(sent, pairs):
    data = []
    pairs = [item for item in pairs if item[sim_str] >= similar_threshold]
    if not pairs:
        return []
    doc = nlp(sent)
    if len(doc) > max_sentence_length or len(doc) < min_sentence_length:
        return []
    for item in pairs:
        # Calculate calculate dependency coverage
        temp_data = collect_sub_dep_path(doc, item[kw1_str], item[kw2_str])
        data.extend(temp_data)
    return data


def reverse_path(path:str):
    path = path.split()
    r_path = ' '.join(['i_' + token if token[:2] != 'i_' else token[2:] for token in reversed(path)])
    return r_path


def gen_pattern(path:str):
    if 'i_nsubj' not in path:
        path = reverse_path(path)
    path = path.split()
    path_ = []
    # Check for 'prep prep'
    for token_idx, token in enumerate(path):
        if 'appos' in token or 'conj' in token:
            continue
        if token_idx > 0:
            if token == 'prep' and path[token_idx - 1] == 'prep':
                continue
        path_.append(token)
    return ' '.join(path_)


def process_line(sent:str, tups:List[Tuple[float, str, str]], sent_note:str, processor):
    
    pairs = [{kw1_str:gen_kw_from_wiki_ent(tup[1], False), kw2_str:gen_kw_from_wiki_ent(tup[2], False), sim_str:tup[0], sent_str:sent_note, kw1_ent_str:tup[1], kw2_ent_str:tup[2]} for tup in tups]
    return processor(sent, pairs)
    
    
def process_list(sents:List[str], pairs_list:List[str], processor):
    data = []
    for line_idx, pairs in enumerate(tqdm.tqdm(pairs_list)):
        if not pairs:
            continue
        pairs = pairs.split('\t')
        tups = [eval(pair) for pair in pairs]
        data.extend(process_line(sents[line_idx], tups, sents[line_idx], processor))
    return data


def process_file(save_sent_file:str, save_pair_file:str, processor, posfix:str='.dat'):
    # Build test data
    with open(save_sent_file) as f_in:
        sents = f_in.read().split('\n')
    with open(save_pair_file) as f_in:
        cooccurs = f_in.read().split('\n')
    data = []
    for line_idx, pairs in enumerate(cooccurs):
        if not pairs:
            continue
        pairs = pairs.split('\t')
        tups = [eval(pair) for pair in pairs]
        sent_note = line2note(save_sent_file, line_idx, posfix=posfix)
        data.extend(process_line(sents[line_idx], tups, sent_note, processor))
    return data


record_columns = feature_columns + [pattern_freq_str, score_str]


def filter_unrelated_from_df(df:pd.DataFrame, similar_threshold):
    return df[df[sim_str] >= similar_threshold]


def cal_freq_from_path(path:str, c:Counter, log_max_cnt:float):
    cnt = c.get(path)
    cnt = (cnt if cnt else 0.5) + 1
    return np.log(cnt) / log_max_cnt


def load_pattern_freq(path_pattern_count_file_:str):
    c:Counter = my_read_pickle(path_pattern_count_file_)
    max_cnt = c.most_common(1)[0][1]
    log_max_cnt = np.log(max_cnt+1)
    return c, log_max_cnt


def cal_freq_from_df(df:pd.DataFrame, c:Counter, log_max_cnt:float):
    return df.assign(pattern_freq = df.apply(lambda x: cal_freq_from_path(x[pattern_str], c, log_max_cnt), axis=1))


def cal_score_from_df(df:pd.DataFrame):
    
    def cal_score(pattern_freq:float, dep_coverage:float):
        return 2 / ((1/pattern_freq)+(1/dep_coverage))

    sub_df = df.assign(score = df.apply(lambda x: cal_score(x[pattern_freq_str], x[dep_coverage_str]), axis=1))
    return sub_df


def get_entity_page(ent:str):
    lines = []
    for f in save_title_files:
        found = False
        with open(f) as f_in:
            for line_idx, line in enumerate(f_in):
                if line.strip() == ent:
                    if not found:
                        found = True
                    lines.append(line2note(f, line_idx, '_ti.dat'))
                else:
                    if found:
                        return lines
    return []


def find_triangles(graph:nx.Graph, node:str):
    triangles = set()
    neighbors = set(graph.neighbors(node))
    for neighbor in neighbors:
        second_neighbors = set(graph.neighbors(neighbor))
        inter_neighbors = neighbors & second_neighbors
        for third_neighbor in inter_neighbors:
            triangles.add(frozenset((node, neighbor, third_neighbor)))
    return triangles


def find_path_between_pair(graph:nx.Graph, first_node:str, second_node:str, hop_num:int=1):
    first_neighbors = set(graph.neighbors(first_node))
    first_neighbors.remove(second_node)
    second_neighbors = set(graph.neighbors(second_node))
    second_neighbors.remove(first_node)
    if hop_num == 1:
        return [[first_node, node, second_node] for node in (first_neighbors & second_neighbors)]
    else:
        return [[first_node, first_neighbor, second_neighbor, second_node] for first_neighbor in first_neighbors for second_neighbor in second_neighbors if graph.has_edge(first_neighbor, second_neighbor)]
    # else:
    #     possible_paths = SimpleQueue()
    #     possible_paths.put([first_node])
    #     ret = []
    #     for i in range(hop_num+1):
    #         path_num = possible_paths.qsize()
    #         for path_idx in range(path_num):
    #             path = possible_paths.get()
    #             for neighbor in graph.neighbors(path[-1]):
    #                 if neighbor not in path:
    #                     new_path = path + [neighbor]
    #                     if neighbor == second_node and len(new_path) > 2:
    #                         ret.append(new_path)
    #                         if max_path_num > 0 and len(ret) >= max_path_num:
    #                             return ret
    #                     else:
    #                         possible_paths.put(new_path)
    #     return ret


def find_all_triangles(graph:nx.Graph):
    return set(frozenset([n,nbr,nbr2]) for n in tqdm.tqdm(graph) for nbr, nbr2 in itertools.combinations(graph[n],2) if nbr in graph[nbr2])


def generate_sample(graph:nx.Graph, ent1:str, ent2:str, max_hop_num:int=2, min_path_num:int=5):
    target = graph.get_edge_data(ent1, ent2)
    if not target:
        return None
    hop_num = 1
    triples = []
    target_note = target['note']
    while hop_num<=max_hop_num:
        paths = find_path_between_pair(graph, ent1, ent2, hop_num=hop_num)
        paths = [{'path' : path} for path in paths]
        for item in paths:
            path = item['path']
            temp_sum = 0
            for i in range(len(path)-1):
                temp_sum += 1 / graph.get_edge_data(path[i], path[i+1])['score']
            item['score'] = (len(path)-1) / temp_sum
        paths.sort(key=lambda x: x['score'], reverse=True)
        for item in paths:
            path = item['path']
            bad_path = False
            temp_path = []
            for i in range(len(path)-1):
                tri = {'e1' : path[i], 'e2' : path[i+1], 'pid' : len(triples)}
                tri.update(graph.get_edge_data(path[i], path[i+1]))
                temp_path.append(tri)
                if tri['note'] == target_note:
                    bad_path = True
                    break
            if not bad_path:
                triples.append(temp_path)
            if len(triples) >= min_path_num:
                break
        if len(triples) >= min_path_num:
            break
        hop_num += 1
    if len(triples) < min_path_num:
        return None
    entity = set()
    source = set()
    for path in triples:
        source.update([tri['note'] for tri in path])
        entity.update([tri['e1'] for tri in path])
        entity.update([tri['e2'] for tri in path])
    entity = list(entity)
    source = list(source)
    for path in triples:
        for tri in path:
            tri['sent'] = source.index(tri['note'])
            tri['e1'] = entity.index(tri['e1'])
            tri['e2'] = entity.index(tri['e2'])
            tri.pop('note')
    source = [note2line(note).strip() for note in source]
    return {'pair' : (ent1, ent2), 
            'entity' : entity, 
            'target' : note2line(target_note).strip(), 
            'source' : source, 
            'triple' : triples}


def generate_second_level_sample(sample:dict):
    second_level_sample = {}
    second_level_sample['key pair'] = sample['pair']
    second_level_sample['target'] = sample['target']
    second_level_sample['sources'] = []
    sources = [nlp(sent) for sent in sample['source']]
    for t in sample['triple']:
        ent1_idx, ent2_idx, sent_idx, kw1_span, kw2_span = t
        kw1_span, kw2_span = eval(kw1_span), eval(kw2_span)
        if kw1_span[0] > kw2_span[0]:
            kw1_span, kw2_span = kw2_span, kw1_span
        doc = sources[sent_idx]
        i = 0
        m = {}
        i2s = {}
        kw1_i, kw2_i = 0, 0
        for j in range(len(doc)):
            m[j] = i
            if j == kw1_span[0]:
                kw1_i = i
                i2s[i] = doc[kw1_span[0]:kw1_span[1]+1].text
            elif j == kw2_span[0]:
                kw2_i = i
                i2s[i] = doc[kw2_span[0]:kw2_span[1]+1].text
            elif i not in i2s:
                i2s[i] = doc[j].text
            if (j < kw1_span[0] or j >= kw1_span[1]) and (j < kw2_span[0] or j >= kw2_span[1]):
                i += 1
        g = []
        tokenized_sent = [[] for _ in range(i)]
        for tok in doc:
            head_idx = m[tok.i]
            tokenized_sent[head_idx].append(tok.text)
            for child in tok.children:
                tail_idx = m[child.i]
                if head_idx != tail_idx:
                    g.extend([(head_idx, tail_idx, child.dep_), (tail_idx, head_idx, 'i_'+child.dep_)])
        tokenized_sent = [' '.join(p) for p in tokenized_sent]
        one_sentence_graph = {'pair' : (kw1_i, kw2_i), 'sent' : tokenized_sent, 'graph' : g}
        second_level_sample['sources'].append(one_sentence_graph)
    return second_level_sample


def sample_to_neo4j(sample:dict):
    cmd = ['MATCH (n) DETACH DELETE (n);']
    create_cmd = []
    match_cmd = []
    for ent in sample['pair']:
        create_cmd.append('CREATE (:ENT:TARGET {ent:"%s"});' % ent)
    for ent in sample['entity']:
        if ent not in sample['pair']:
            create_cmd.append('CREATE (:ENT:INTERMEDIA {ent:"%s"});' % ent)
    for path in sample['triple']:
        for tri in path:
            ent1 = sample['entity'][tri['e1']]
            ent2 = sample['entity'][tri['e2']]
            sent = sample['source'][tri['sent']]
            score = tri['score']
            match_cmd.append('MATCH (ent1:ENT {ent:"%s"}), (ent2:ENT {ent:"%s"}) CREATE (ent1)-[:Sent {sent:"%s", pair:"%s <-> %s", score:%.3f}]->(ent2);' % (ent1, ent2, sent.replace('"', '\\"'), ent1, ent2, score))
    match_cmd = list(set(match_cmd))
    cmd.extend(create_cmd)
    cmd.extend(match_cmd)
    cmd.append('MATCH (ent1:ENT {ent:"%s"}), (ent2:ENT {ent:"%s"}) CREATE (ent1)-[:OUT {sent:"%s", pair:"%s <-> %s"}]->(ent2);' % (*sample['pair'], sample['target'].replace('"', '\\"'), *sample['pair']))
    print('\n'.join(cmd))


modifier_dependencies = {'acl', 'advcl', 'advmod', 'amod', 'det', 'mark', 'meta', 'neg', 'nn', 'nmod', 'npmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl',
                         'appos', 'aux', 'auxpass', 'compound', 'cop', 'ccomp', 'xcomp', 'expl', 'punct', 'nsubj', 'csubj', 'csubjpass', 'dobj', 'iobj', 'obj', 'pobj'}

def expand_dependency_info_from_tree(doc, path:np.ndarray):
    dep_path:list = (np.arange(*path.shape)[path!=0]).tolist()
    for element in dep_path:
        if doc[element].dep_ == 'conj':
            path[doc[element].head.i] = 0
    modifiers = []
    for element in dep_path:
        for child in doc[element].children:
            if path[element] == 0 and child.dep_ == 'compound':
                continue
            if path[child.i] == 0 and child.dep_ in modifier_dependencies:
                path[child.i] = 1
                modifiers.append(child.i)
    while len(modifiers) > 0:
        modifier = modifiers.pop(0)
        for child in doc[modifier].children:
            if path[child.i] == 0:
                path[child.i] = 1
                modifiers.append(child.i)


def get_path(doc, kw1_steps:List[int], kw2_steps:List[int]):
    path_tokens = []
    for step in kw1_steps:
        path_tokens.append('i_' + doc[step].dep_)
    kw2_steps.reverse()
    for step in kw2_steps:
        path_tokens.append(doc[step].dep_)
    return ' '.join(path_tokens)


def find_dependency_info_from_tree(doc, kw1, kw2):
    # Find roots of the spans
    idx1 = find_root_in_span(kw1)
    idx2 = find_root_in_span(kw2)
    kw1_front, kw1_end = kw1[0].i, kw1[-1].i
    kw2_front, kw2_end = kw2[0].i, kw2[-1].i
    branch = np.zeros(len(doc))
    kw1_steps = []
    kw2_steps = []
    path_found = False
    
    i = idx1
    while branch[i] == 0:
        branch[i] = 1
        kw1_steps.append(i)
        i = doc[i].head.i
        if i >= kw2_front and i <= kw2_end:
            # kw2 is above kw1
            path_found = True
            break
        
    if not path_found:
        i = idx2
        while branch[i] != 1:
            branch[i] = 2
            kw2_steps.append(i)
            if i == doc[i].head.i:
                return [], [], np.array([])
            
            i = doc[i].head.i
            if i >= kw1_front and i <= kw1_end:
                # kw1 is above kw2
                branch[branch != 2] = 0
                kw1_steps = []
                path_found = True
                break
    
    if not path_found:
        # kw1 and kw2 are on two sides, i is their joint
        break_point = kw1_steps.index(i)
        branch[kw1_steps[break_point+1 : ]] = 0
        kw1_steps = kw1_steps[:break_point] # Note that we remain the joint node in the branch, but we don't include joint point in kw1_steps and kw2_steps
                                            # this is because the joint node is part of the path and we need the modification information from it, 
                                            # but we don't care about its dependency
    branch[branch != 0] = 1
    branch[kw1_front : kw1_end+1] = 1
    branch[kw2_front : kw2_end+1] = 1
    return kw1_steps, kw2_steps, branch


def collect_sub_dependency_path(doc, branch:np.ndarray):
    paths = []
    dep_path:list = (np.arange(*branch.shape)[branch!=0]).tolist()
    for token_id in dep_path:
        temp_paths = [(token_id, child.dep_, child.i) for child in doc[token_id].children if branch[child.i] == 0]
        while len(temp_paths) > 0:
            item  = temp_paths.pop()
            paths.append(item)
            temp_paths.extend([(item[0], item[1] + ' ' + child.dep_, child.i) for child in doc[item[2]].children if branch[child.i] == 0])
    return paths


def informativeness_demo(sent:str, kw1:str, kw2:str, fp:FeatureProcess):
    doc = nlp(sent)
    kw1_span = find_span(doc, kw1, True, True)[0]
    kw2_span = find_span(doc, kw2, True, True)[0]
    kw1_steps, kw2_steps, path = find_dependency_info_from_tree(doc, kw1_span, kw2_span)
    fp.expand_dependency_info_from_tree(doc, path)
    context = []
    temp = []
    for i, checked in enumerate(path):
        if checked:
            temp.append(doc[i].text)
        else:
            if temp:
                context.append(' '.join(temp))
                temp = []
    if temp:
        context.append(' '.join(temp))
    return pd.DataFrame({i:[doc[i].text, np.round(path[i], 3)] for i in range(len(doc))})


def generate_single_sent_graph(graph:nx.Graph, score_threshold:float):
        single_sent_g = nx.Graph()
        for edge in tqdm.tqdm(graph.edges):
            data = graph.get_edge_data(*edge)
            sim = data['sim']
            data = data['data']
            best_score, best_note, best_span, best_dep_coverage, best_pattern_freq = data[0]
            for score, note, span, dep_coverage, pattern_freq in data:
                if score > best_score:
                    best_score, best_note, best_span, best_dep_coverage, best_pattern_freq = score, note, span, dep_coverage, pattern_freq
            if best_score >= score_threshold:
                single_sent_g.add_edge(*edge, score=best_score, note=best_note, sim=sim, span=best_span, significant=best_dep_coverage, explict=best_pattern_freq)
        return single_sent_g
        
        
if __name__ == '__main__':
    # Generate the save dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for save_dir in save_sub_folders:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    if sys.argv[1] == 'collect_sent_and_cooccur':
        
        wiki_sent_pair = [(wiki_files[i], save_sent_files[i], save_cooccur_files[i], save_title_files[i]) for i in range(len(wiki_files))]
        output = p.run(get_sentence, wiki_sent_pair)
        
        # Collect wikipedia entities
        wikipedia_entity = set()
        for f in tqdm.tqdm(save_title_files):
            with open(f) as f_in:
                wikipedia_entity.update(f_in.read().split('\n'))
        print(len(wikipedia_entity))
        my_write(wikipedia_entity_file, list(wikipedia_entity))
        
    
    elif sys.argv[1] == 'correct_mapping_in_cooccur':
        
        # Load wikipedia2vec
        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)
    
        # Load wikipedia entities
        with open(wikipedia_entity_file) as f_in:
            wikipedia_entity = set(f_in.read().split('\n'))
    
        # Generate lower-cased entity to original entity mapping
        print('Generate lower-cased entity to original entity mapping')
        wikipedia_entity_low2orig_map = {}
        for ent in wikipedia_entity:
            ent_low = ent.lower()
            if ent_low not in wikipedia_entity_low2orig_map:
                wikipedia_entity_low2orig_map[ent_low] = []
            wikipedia_entity_low2orig_map[ent_low].append(ent)

        # Correct mapping
        print('Correct mapping')
        for i in tqdm.tqdm(range(len(save_cooccur_files))):
            with open(save_cooccur_files[i]) as f_in:
                new_file_lines = []
                for line_idx, line in enumerate(f_in):
                    line = line.strip()
                    entities = line.split('\t')
                    new_entities = []
                    for ent in entities:
                        if ent in wikipedia_entity:
                            new_entities.append(ent)
                        else:
                            ent_low = ent.lower()
                            if ent_low in wikipedia_entity_low2orig_map:
                                candidates = wikipedia_entity_low2orig_map[ent_low]
                                if len(candidates) == 1:
                                    new_entities.append(candidates[0])
                                else:
                                    note = line2note(save_cooccur_files[i], line_idx, '_co.dat')
                                    page_title = note2line(note, '_ti.dat').strip()
                                    try:
                                        page_ent_vec = w2vec.get_entity_vector(page_title)
                                    except:
                                        continue
                                    most_similar_idx, most_similar_val = -1, -1
                                    for candidate_idx, candidate_ent in enumerate(candidates):
                                        try:
                                            candidate_vec = w2vec.get_entity_vector(candidate_ent)
                                        except:
                                            continue
                                        similar_val = cosine_similarity(page_ent_vec.reshape(1, -1), candidate_vec.reshape(1, -1))[0,0]
                                        if similar_val > most_similar_val:
                                            most_similar_val = similar_val
                                            most_similar_idx = candidate_idx
                                    if most_similar_idx >= 0:
                                        new_entities.append(candidates[most_similar_idx])
                    new_file_lines.append('\t'.join(new_entities))
                my_write(save_cooccur__files[i], new_file_lines)
                
                
    elif sys.argv[1] == 'cal_cooccur_similarity':
        
        # Load wikipedia2vec
        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)
        
        for f_id, save_cooccur__file in enumerate(tqdm.tqdm(save_cooccur__files)):
            with open(save_cooccur__file) as f_in:
                cooccurs = f_in.read().split('\n')
            data = []
            for line in cooccurs:
                ents = line.split('\t')
                if len(ents) <= 1:
                    data.append('')
                else:
                    temp_data = []
                    valid_entities = []
                    matrix = []
                    for ent in ents:
                        try:
                            vec = w2vec.get_entity_vector(ent)
                            matrix.append(vec)
                            valid_entities.append(ent)
                        except:
                            pass
                    certain_len = len(valid_entities)
                    if certain_len >= 2:
                        # Collect pairs between certain entities
                        matrix = np.array(matrix)
                        result = cosine_similarity(matrix, matrix)
                        for i in range(certain_len):
                            for j in range(i+1, certain_len):
                                tup = (float(result[i, j]), valid_entities[i], valid_entities[j])
                                temp_data.append(str(tup))
                    data.append('\t'.join(temp_data))
            my_write(save_pair_files[f_id], data)
            
            
    elif sys.argv[1] == 'collect_ent_occur_from_cooccur':
        
        def collect_ent_occur_from_cooccur(files:list, keyword_dict:dict):
            for file in tqdm.tqdm(files):
                with open(file) as f_in:
                    for i, line in enumerate(f_in):
                        entities = line.strip().split('\t')
                        if not entities:
                            continue
                        note = line2note(file, i, '_co_.dat')
                        for ent in entities:
                            if ent not in keyword_dict:
                                keyword_dict[ent] = set()
                            keyword_dict[ent].add(note)
                
        entity_occur = {}
        collect_ent_occur_from_cooccur(save_cooccur__files, entity_occur)
        my_write_pickle(entity_occur_from_cooccur_file, entity_occur)
        
        
    elif sys.argv[1] == 'collect_sub_path_pattern_freq':
        
        sents = []
        pairs_list = []
        for file_idx in range(50):
            with open(save_sent_files[file_idx]) as f_in:
                sents.extend(f_in.read().split('\n'))
            with open(save_pair_files[file_idx]) as f_in:
                pairs_list.extend(f_in.read().split('\n'))
        
        c = Counter(process_list(sents, pairs_list, batched_collect_sub_dep_path))
        my_write_pickle(sub_path_pattern_count_file, c)
        
        
    elif sys.argv[1] == 'collect_pattern_freq':
        
        sents = []
        pairs_list = []
        for file_idx in range(50):
            with open(save_sent_files[file_idx]) as f_in:
                sents.extend(f_in.read().split('\n'))
            with open(save_pair_files[file_idx]) as f_in:
                pairs_list.extend(f_in.read().split('\n'))
        
        # with open('data/extract_wiki/wiki_sent_collect/BE/wiki_15.dat') as f_in:
        #     sents.extend(f_in.read().split('\n'))
        # with open('data/extract_wiki/wiki_sent_collect/BE/wiki_15_pr.dat') as f_in:
        #     pairs_list.extend(f_in.read().split('\n'))
        fp = FeatureProcess(sub_path_pattern_count_file)
        
        wiki_path_test_df = pd.DataFrame(process_list(sents, pairs_list, fp.batched_feature_process))
        wiki_path_test_df = filter_unrelated_from_df(wiki_path_test_df, similar_threshold)
        wiki_path_test_df.to_csv(path_test_file, sep='\t', columns=feature_columns, index=False)
        print(len(wiki_path_test_df))
        c = Counter(wiki_path_test_df['pattern'].to_list())
        my_write_pickle(path_pattern_count_file, c)
        
        
    elif sys.argv[1] == 'collect_dataset':
        
        c, log_max_cnt = load_pattern_freq(path_pattern_count_file)
        fp = FeatureProcess(sub_path_pattern_count_file)
        
        def collect_dataset(save_sent_file:str, save_pair_file:str, save_selected_file, posfix:str='.dat'):
            pairs = process_file(save_sent_file, save_pair_file, fp.batched_feature_process, posfix)
            pairs = [item for item in pairs if 'nsubj' in item[dep_path_str]]
            data = pd.DataFrame(pairs)
            # data = filter_unrelated_from_df(data, similar_threshold)
            data = cal_freq_from_df(data, c, log_max_cnt)
            data = cal_score_from_df(data)
            data.to_csv(save_selected_file, columns=record_columns, sep='\t', index=False)

        input_list = [(save_sent_files[i], save_pair_files[i], save_selected_files[i]) for i in range(len(save_sent_files))]
        _ = p.run(collect_dataset, input_list)
        
        # for i in range(2):
        #     collect_dataset(save_sent_files[i], save_pair_files[i], save_selected_files[i])
        
        
    elif sys.argv[1] == 'generate_graph':
        
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        def generate_graph(files:list, sim_threshold:float):
            g = nx.Graph()
            for file in tqdm.tqdm(files):
                with open(file) as f_in:
                    head_idx, tail_idx, sent_idx, score_idx, sim_idx = -1, -1, -1, -1, -1
                    for i, line in enumerate(csv.reader(f_in, delimiter='\t')):
                        if i == 0:
                            head_idx, tail_idx, sent_idx, score_idx, sim_idx, head_span_idx, tail_span_idx, dep_coverage_idx, pattern_freq_idx = line.index(kw1_ent_str), line.index(kw2_ent_str), line.index(sent_str), line.index(score_str), line.index(sim_str), line.index(kw1_span_str), line.index(kw2_span_str), line.index(dep_coverage_str), line.index(pattern_freq_str)
                            continue
                        sim = float(line[sim_idx])
                        if sim < sim_threshold:
                            continue
                        if not g.has_edge(line[head_idx], line[tail_idx]):
                            g.add_edge(line[head_idx], line[tail_idx], sim = sim, data = [(float(line[score_idx]), line[sent_idx], (line[head_span_idx], line[tail_span_idx]), float(line[dep_coverage_idx]), float(line[pattern_freq_idx]))])
                        else:
                            data = g.get_edge_data(line[head_idx], line[tail_idx])
                            data['data'].append((float(line[score_idx]), line[sent_idx], (line[head_span_idx], line[tail_span_idx]), float(line[dep_coverage_idx]), float(line[pattern_freq_idx])))
            return g

        graph = generate_graph(save_selected_files, similar_threshold)
        my_write_pickle(graph_file, graph)
        
    
    elif sys.argv[1] == 'generate_single_sent_graph':
        
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        graph = my_read_pickle(graph_file)
        single_sent_g = generate_single_sent_graph(graph, 0.6)
        my_write_pickle(single_sent_graph_file, single_sent_g)
            
        
    elif sys.argv[1] == 'collect_score_function_eval_dataset':
        
        graph:nx.Graph = my_read_pickle(graph_file)
        entity_occur_from_cooccur = my_read_pickle(entity_occur_from_cooccur_file)
        test_pairs = []
        for pair in random.sample(graph.edges, 500):
            ent1, ent2 = pair
            occur1 = entity_occur_from_cooccur.get(ent1)
            occur2 = entity_occur_from_cooccur.get(ent2)
            if occur1 is None or occur2 is None:
                continue
            intersect = occur1 & occur2
            if len(intersect) < 5:
                continue
            selected_sent = random.choice(graph.get_edge_data(*pair)['data'])[1]
            # print(selected_sent)
            temp_intersect = intersect - {selected_sent}
            random_sent = random.sample(temp_intersect, 1)[0]
            notes = [random_sent, selected_sent]
            random.shuffle(notes)
            for note in notes:
                test_pairs.append({'entity 1' : ent1, 'entity 2' : ent2, 'sentence' : note2line(note).strip(), 'score' : 0})
            if len(test_pairs) >= 100:
                break
        test_data = pd.DataFrame(test_pairs)
        test_data.to_csv('test.tsv', sep='\t', index=False)
        
        
    elif sys.argv[1] == 'recalculate_score':
        
        for file in tqdm.tqdm(save_selected_files):
            with open(file) as f_in:
                data = pd.read_csv(f_in, sep='\t')
                data = cal_score_from_df(data)
            data.to_csv(file, columns=record_columns, sep='\t', index=False)
            
    
    elif sys.argv[1] == 'collect_sample_from_single_sent_graph':
        
        context_sent_score_threshold = 0.6
        target_edges = list(my_read_pickle(single_sent_graph_file).edges)
        source_sent_g = my_read_pickle(graph_file)
        source_sent_g:nx.Graph = generate_single_sent_graph(source_sent_g, context_sent_score_threshold)
        
        sample_num = -1
        samples = []
        print(len(target_edges))
        edge_count = 0
        for edge_idx, edge in enumerate(tqdm.tqdm(target_edges)):
            edge_count += 1
            sample = generate_sample(source_sent_g, edge[0], edge[1])
            if sample:
                samples.append(sample)
            if (sample_num > 0) and (len(samples) >= sample_num):
                break
        print(len(samples) * 1.0 / edge_count)
    
        with open('dataset_level_1.json', 'w') as f_out:
            json.dump(samples, f_out)
            print(len(samples))
            
    
    elif sys.argv[1] == 'collect_second_level_sample':
        
        with open('dataset_level_1.json') as f_in:
            samples = json.load(f_in)
        
        second_level_samples = [generate_second_level_sample(sample) for sample in tqdm.tqdm(samples)]
        with open('dataset_level_2.json', 'w') as f_out:
            json.dump(second_level_samples, f_out)
            print(len(second_level_samples))
        
    elif sys.argv[1] == 'collect_triangles_from_graph':
        
        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)
            
        graph = my_read_pickle(single_sent_graph_file)
        edges = [edge for edge in tqdm.tqdm(graph.edges) if graph.get_edge_data(*edge)['score'] > 0.65]
        filtered_graph = graph.edge_subgraph(edges)
        nodes = []
        for node in tqdm.tqdm(filtered_graph):
            ent = w2vec.get_entity(node)
            if ent is None:
                continue
            if ent.count >= 20:
                nodes.append(node)
        filtered_graph = filtered_graph.subgraph(nodes)
        triangle_set = find_all_triangles(filtered_graph)
        my_write_pickle('data/extract_wiki/triangles.pickle', triangle_set)