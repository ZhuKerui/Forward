# python extract_wiki.py collect_sent_and_cooccur
# python extract_wiki.py collect_ent_occur_from_cooccur
# python extract_wiki.py collect_pattern_freq
# python extract_wiki.py collect_dataset
# python extract_wiki.py collect_score_function_eval_dataset
# python extract_wiki.py generate_graph
# python extract_wiki.py generate_single_sent_graph
# python extract_wiki.py collect_one_hop_sample_from_single_sent_graph
# python extract_wiki.py collect_second_level_sample

import re
import os
from collections import Counter
from queue import SimpleQueue
import json
import random
from nltk.tokenize import sent_tokenize
import tqdm
import csv
import networkx as nx
import pandas as pd
from urllib.parse import unquote
import sys
import linecache
import numpy as np
from typing import List
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
save_selected_files = []
save_title_files = []

for i in range(len(wiki_sub_folders)):
    files = [f for f in os.listdir(wiki_sub_folders[i])]
    wiki_files += [os.path.join(wiki_sub_folders[i], f) for f in files]
    save_sent_files += [os.path.join(save_sub_folders[i], f+'.dat') for f in files]
    save_cooccur_files += [os.path.join(save_sub_folders[i], f+'_co.dat') for f in files]
    save_cooccur__files += [os.path.join(save_sub_folders[i], f+'_co_.dat') for f in files]
    save_selected_files += [os.path.join(save_sub_folders[i], f+'_se.dat') for f in files]
    save_title_files += [os.path.join(save_sub_folders[i], f+'_ti.dat') for f in files]

wikipedia_entity_file = 'data/extract_wiki/wikipedia_entity.tsv'

w2vec_dump_file = 'data/extract_wiki/enwiki_20180420_win10_100d.pkl.bz2'
w2vec_keyword2idx_file = 'data/extract_wiki/w2vec_keyword2idx.pickle'

entity_occur_from_cooccur_file = 'data/extract_wiki/entity_occur_from_cooccur.pickle'
path_test_file = 'data/extract_wiki/wiki_sent_test/path_test.tsv'
path_pattern_count_file = 'data/extract_wiki/path_pattern.pickle'

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
__const_num = 0.5
__kw1 = 'kw1'
__kw2 = 'kw2'
__kw1_ent = 'kw1_ent'
__kw2_ent = 'kw2_ent'
__sent = 'sent'
__kw1_span = 'kw1_span'
__kw2_span = 'kw2_span'
__dep_path = 'dep_path'
__dep_coverage = 'dep_coverage'
__sim = 'sim'
__pattern = 'pattern'
__pattern_freq = 'pattern_freq'
__score = 'score'
__similar_threshold = 0.5
__max_sentence_length = 50
__min_sentence_length = 5

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
feature_columns=[__sim, __kw1, __kw1_span, __kw1_ent, __kw2, __kw2_span, __kw2_ent, __sent, __dep_path, __pattern, __dep_coverage]

def get_phrase_full_span(doc, phrase_span):
    phrase_right_most = get_back(doc, phrase_span[-1].i)
    phrase_left_most = min(get_ahead(doc, phrase_right_most), get_ahead(doc, phrase_span[0].i), phrase_span[0].i)
    return (phrase_left_most, phrase_right_most)

def cal_kw_recall(kw:str, full_phrase:str):
    return np.log(kw.count(' ') + 1 + __const_num) / np.log(full_phrase.count(' ') + 1 + __const_num)

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
            data.append((kw1_span, kw2_span, branch, pattern))
    return data


def feature_process(doc, kw1:str, kw2:str)->List[dict]:
    data = []
    for kw1_span, kw2_span, branch, pattern in sentence_decompose(doc, kw1, kw2):
        expand_dependency_info_from_tree(doc, branch)
        data.append({__kw1_span : (kw1_span[0].i, kw1_span[-1].i),
                     __kw2_span : (kw2_span[0].i, kw2_span[-1].i),
                     __pattern : pattern, 
                     __dep_coverage : branch.mean()
                    })
    return data


def collect_sub_dep_path(doc, kw1:str, kw2:str)->List[dict]:
    data = []
    for kw1_span, kw2_span, branch, pattern in sentence_decompose(doc, kw1, kw2):
        ans = [str(item[1]) for item in collect_sub_dependency_path(doc, branch)]
        ans = [item.replace('compound', '').replace('conj', '').replace('appos', '') for item in ans if item != 'punct']
        data.extend(ans)
    return data


def batched_feature_process(doc, pairs):
    data = []
    for item in pairs:
        # Calculate calculate dependency coverage
        temp_data = feature_process(doc, item[__kw1], item[__kw2])
        for d in temp_data:
            d.update(item)
        data.extend(temp_data)
    return data


def batched_collect_sub_dep_path(doc, pairs):
    data = []
    for item in pairs:
        # Calculate calculate dependency coverage
        temp_data = collect_sub_dep_path(doc, item[__kw1], item[__kw2])
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


def process_line(sent:str, ents:list, w2vec:Wikipedia2Vec, sent_note:str, processor):
    co_kws = []
    matrix = []
    for ent in ents:
        try:
            vec = w2vec.get_entity_vector(ent)
            matrix.append(vec)
            co_kws.append(gen_kw_from_wiki_ent(ent))
        except:
            pass
    certain_len = len(co_kws)
    if certain_len <= 1:
        return []
    
    pairs = []
    # Collect pairs between certain entities
    matrix = np.array(matrix)
    result = cosine_similarity(matrix, matrix)
    for i in range(certain_len):
        for j in range(i+1, certain_len):
            pairs.append({__kw1:co_kws[i], __kw2:co_kws[j], __sim:float(result[i, j]), __sent:sent_note, __kw1_ent:ents[i], __kw2_ent:ents[j]})
    doc = nlp(sent)
    if len(doc) > __max_sentence_length or len(doc) < __min_sentence_length:
        return []
    return processor(doc, pairs)
    
    
def process_list(sents:List[str], cooccurs:List[str], w2vec:Wikipedia2Vec, processor):
    data = []
    for line_idx, line in enumerate(tqdm.tqdm(cooccurs)):
        line = line.split('\t')
        if len(line) <= 1:
            continue
        data.extend(process_line(sents[line_idx], line, w2vec, sents[line_idx], processor))
    return data


def process_file(save_sent_file:str, save_cooccur__file:str, w2vec:Wikipedia2Vec, processor, posfix:str='.dat'):
    # Build test data
    with open(save_sent_file) as f_in:
        sents = f_in.read().split('\n')
    with open(save_cooccur__file) as f_in:
        cooccurs = f_in.read().split('\n')
    data = []
    for line_idx, line in enumerate(cooccurs):
        line = line.split('\t')
        if len(line) <= 1:
            continue
        sent_note = line2note(save_sent_file, line_idx, posfix=posfix)
        data += process_line(sents[line_idx], line, w2vec, sent_note, processor)
    return data


record_columns = feature_columns + [__pattern_freq, __score]


def filter_unrelated_from_df(df:pd.DataFrame, similar_threshold):
    return df[df[__sim] >= similar_threshold]


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
    return df.assign(pattern_freq = df.apply(lambda x: cal_freq_from_path(x[__pattern], c, log_max_cnt), axis=1))


def cal_score_from_df(df:pd.DataFrame):
    
    def cal_score(pattern_freq:float, dep_coverage:float):
        return 2 / ((1/pattern_freq)+(1/dep_coverage))

    sub_df = df.assign(score = df.apply(lambda x: cal_score(x[__pattern_freq], x[__dep_coverage]), axis=1))
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


def find_path_between_pair(graph:nx.Graph, first_node:str, second_node:str, hop_num:int=1, max_path_num:int=-1):
    if hop_num == 1:
        one_hop_nodes = list(set(graph.neighbors(first_node)) & set(graph.neighbors(second_node)))
        ret = [[first_node, node, second_node] for node in one_hop_nodes]
        if max_path_num > 0 and len(ret) > max_path_num:
            ret = ret[:max_path_num]
    else:
        possible_paths = SimpleQueue()
        possible_paths.put([first_node])
        ret = []
        for i in range(hop_num+1):
            path_num = possible_paths.qsize()
            for path_idx in range(path_num):
                path = possible_paths.get()
                for neighbor in graph.neighbors(path[-1]):
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        if neighbor == second_node:
                            ret.append(new_path)
                            if max_path_num > 0 and len(ret) >= max_path_num:
                                return ret
                        else:
                            possible_paths.put(new_path)
    return ret


def find_all_triangles(graph:nx.Graph):
    return set(frozenset([n,nbr,nbr2]) for n in tqdm.tqdm(graph) for nbr, nbr2 in itertools.combinations(graph[n],2) if nbr in graph[nbr2])


def generate_sample(graph:nx.Graph, ent1:str, ent2:str, hop_num:int=1, max_path_num:int=-1):
    target = graph.get_edge_data(ent1, ent2) if graph.has_edge(ent1, ent2) else None
    if not target:
        return None
    paths = find_path_between_pair(graph, ent1, ent2, hop_num=hop_num, max_path_num=max_path_num)
    if not paths:
        return None
    entity = set()
    for path in paths:
        entity.update(path)
    entity = list(entity)
    triples = []
    target:list = target['data']
    target.sort(key=lambda x: x[0], reverse=True)
    target_note = target[0][1]
    for path in paths:
        idx_path = [entity.index(i) for i in path]
        sents = [(
            graph.get_edge_data(path[i], path[i+1])['data'],
            (idx_path[i], idx_path[i+1])
            )for i in range(len(path)-1)]
        temp_triples = []
        sent_in_use = set([target_note])
        sents.sort(key=lambda x: len(x[0]))
        note_found = False
        for sent_item in sents:
            sent_item[0].sort(key=lambda x: x[0], reverse=True)
            note_found = False
            for note_item in sent_item[0]:
                if note_item[1] not in sent_in_use:
                    sent_in_use.add(note_item[1])
                    temp_triples.append({'e1' : sent_item[1][0],
                                         'e2' : sent_item[1][1],
                                         'sent' : note_item[1], 
                                         'score' : note_item[0], 
                                         'span' : note_item[2],
                                         'pid' : len(triples)})
                    note_found = True
                    break
            if not note_found:
                break
        if not note_found:
            continue
        re_ordered_temp_triples = [{} for i in range(len(path)-1)]
        for item in temp_triples:
            re_ordered_temp_triples[idx_path.index(item['e1'])] = item
        triples.append(re_ordered_temp_triples)
    if not triples:
        return None
    source = set()
    for path in triples:
        source.update([item['sent'] for item in path])
    source = list(source)
    for path in triples:
        for item in path:
            item['sent'] = source.index(item['sent'])
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
    for ent in sample['pair']:
        cmd.append('CREATE (:ENT:TARGET {ent:"%s"});' % ent)
    for ent in sample['entity']:
        if ent not in sample['pair']:
            cmd.append('CREATE (:ENT:INTERMEDIA {ent:"%s"});' % ent)
    for path in sample['triple']:
        for tri in path:
            ent1 = sample['entity'][tri['e1']]
            ent2 = sample['entity'][tri['e2']]
            sent = sample['source'][tri['sent']]
            score = tri['score']
            cmd.append('MATCH (ent1:ENT {ent:"%s"}), (ent2:ENT {ent:"%s"}) CREATE (ent1)-[:Sent {sent:"%s", pair:"%s <-> %s", score:%.3f}]->(ent2);' % (ent1, ent2, sent, ent1, ent2, score))
    cmd.append('MATCH (ent1:ENT {ent:"%s"}), (ent2:ENT {ent:"%s"}) CREATE (ent1)-[:OUT {sent:"%s", pair:"%s <-> %s"}]->(ent2);' % (*sample['pair'], sample['target'], *sample['pair']))
    print('\n'.join(cmd))


modifier_dependencies = {'acl', 'advcl', 'advmod', 'amod', 'det', 'mark', 'meta', 'neg', 'nn', 'nmod', 'npmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl'}
adjunctive_dependencies = {'appos', 'aux', 'auxpass', 'compound', 'cop', 'expl', 'punct', 'nsubj', 'dobj'}

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
            if path[child.i] == 0 and (child.dep_ in modifier_dependencies or child.dep_ in adjunctive_dependencies):
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

def informativeness_demo(sent:str, kw1:str, kw2:str):
    doc = nlp(sent)
    kw1_span = find_span(doc, kw1, True, True)[0]
    kw2_span = find_span(doc, kw2, True, True)[0]
    path = find_dependency_info_from_tree(doc, kw1_span, kw2_span)
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
    return context


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
        
        
    elif sys.argv[1] == 'collect_pattern_freq':
        sents = []
        cooccurs = []
        for file_idx in range(16):
            with open(save_sent_files[file_idx]) as f_in:
                sents.extend(f_in.read().split('\n'))
            with open(save_cooccur__files[file_idx]) as f_in:
                cooccurs.extend(f_in.read().split('\n'))
                
        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)
        
        wiki_path_test_df = pd.DataFrame(process_list(sents, cooccurs, w2vec, batched_feature_process))
        wiki_path_test_df = filter_unrelated_from_df(wiki_path_test_df, __similar_threshold)
        wiki_path_test_df.to_csv(path_test_file, sep='\t', columns=feature_columns, index=False)
        print(len(wiki_path_test_df))
        c = Counter(wiki_path_test_df['pattern'].to_list())
        my_write_pickle(path_pattern_count_file, c)
        
        
    elif sys.argv[1] == 'collect_dataset':

        with bz2.open(w2vec_dump_file) as f_in:
            w2vec = Wikipedia2Vec.load(f_in)
        
        c, log_max_cnt = load_pattern_freq(path_pattern_count_file)
        
        def collect_dataset(save_sent_file:str, save_cooccur__file:str, save_selected_file, posfix:str='.dat'):
            pairs = process_file(save_sent_file, save_cooccur__file, w2vec, batched_feature_process, posfix)
            pairs = [item for item in pairs if 'nsubj' in item[__dep_path]]
            data = pd.DataFrame(pairs)
            data = filter_unrelated_from_df(data, __similar_threshold)
            data = cal_freq_from_df(data, c, log_max_cnt)
            data = cal_score_from_df(data)
            data.to_csv(save_selected_file, columns=record_columns, sep='\t', index=False)

        input_list = [(save_sent_files[i], save_cooccur__files[i], save_selected_files[i]) for i in range(len(save_sent_files))]
        _ = p.run(collect_dataset, input_list)
        
        
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
        
        
    elif sys.argv[1] == 'generate_graph':
        
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        def generate_graph(files:list, threshold:float):
            g = nx.Graph()
            for file in tqdm.tqdm(files):
                with open(file) as f_in:
                    head_idx, tail_idx, sent_idx, score_idx, sim_idx = -1, -1, -1, -1, -1
                    for i, line in enumerate(csv.reader(f_in, delimiter='\t')):
                        if i == 0:
                            head_idx, tail_idx, sent_idx, score_idx, sim_idx, head_span_idx, tail_span_idx = line.index(__kw1_ent), line.index(__kw2_ent), line.index(__sent), line.index(__score), line.index(__sim), line.index(__kw1_span), line.index(__kw2_span)
                            continue
                        score = float(line[score_idx])
                        if score < threshold:
                            continue
                        if not g.has_edge(line[head_idx], line[tail_idx]):
                            g.add_edge(line[head_idx], line[tail_idx], sim = float(line[sim_idx]), data = [(score, line[sent_idx], (line[head_span_idx], line[tail_span_idx]))])
                        else:
                            data = g.get_edge_data(line[head_idx], line[tail_idx])
                            data['data'].append((score, line[sent_idx], (line[head_span_idx], line[tail_span_idx])))
            return g

        # graph = generate_graph(save_selected_files, 0.6)
        graph = generate_graph(save_selected_files, 0.3)
        my_write_pickle(graph_file, graph)
            
            
    elif sys.argv[1] == 'generate_single_sent_graph':
        
        # Load keyword occur dict which has occurance record for all keywords in selected sentences
        def generate_single_sent_graph(graph:nx.Graph):
            single_sent_g = nx.Graph()
            for edge in tqdm.tqdm(graph.edges):
                data = graph.get_edge_data(*edge)
                sim = data['sim']
                data = data['data']
                best_score, best_note, best_span = data[0]
                for score, note, span in data:
                    if score > best_score:
                        best_score = score
                        best_note = note
                        best_span = span
                single_sent_g.add_edge(*edge, score=best_score, note=best_note, sim=sim, span=best_span)
            return single_sent_g

        graph = my_read_pickle(graph_file)
            
        single_sent_g = generate_single_sent_graph(graph)
        my_write_pickle(single_sent_graph_file, single_sent_g)
            

    elif sys.argv[1] == 'collect_one_hop_sample_from_single_sent_graph':
        
        single_sent_graph:nx.Graph = my_read_pickle(single_sent_graph_file)
        graph:nx.Graph = my_read_pickle(graph_file)
        
        target_edges = [edge for edge in single_sent_graph.edges if single_sent_graph.get_edge_data(*edge)['score'] > 0.8]
        # target_sent_graph = graph.edge_subgraph(edges)
        
        sample_num = -1
        samples = []
        print(len(target_edges))
        status_bar = tqdm.tqdm(total=min(sample_num, len(target_edges)) if sample_num > 0 else len(target_edges))
        edge_count = 0
        # input_list = [(graph, edge[0], edge[1], 1) for edge in target_edges[:sample_num]]
        for edge_idx, edge in enumerate(target_edges):
            edge_count += 1
            sample = generate_sample(graph, edge[0], edge[1], 1, 10)
            if sample is not None:
                samples.append(sample)
                if edge_idx % 20 == 0:
                    status_bar.update(20)
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