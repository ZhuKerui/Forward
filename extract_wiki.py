# python extract_wiki.py collect_sent_and_cooccur
# python extract_wiki.py correct_mapping_in_cooccur
# python extract_wiki.py cal_cooccur_similarity
# python extract_wiki.py collect_ent_occur_from_cooccur
# python extract_wiki.py collect_subpath_pattern_freq
# python extract_wiki.py collect_pattern_freq
# python extract_wiki.py collect_dataset
# python extract_wiki.py generate_graph
# python extract_wiki.py generate_sent_graph [threshold] [significant/explicit/score]
# python extract_wiki.py generate_random_sent_graph
# python extract_wiki.py collect_sample_from_single_sent_graph
# python extract_wiki.py collect_score_function_eval_dataset
# python extract_wiki.py collect_one_hop_sample_from_single_sent_graph
# python extract_wiki.py collect_second_level_sample

from copy import deepcopy
import re
import os
from collections import Counter
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

from tools.BasicUtils import MyMultiProcessing, my_write, my_read_pickle, my_write_pickle
from tools.TextProcessing import (remove_brackets, find_root_in_span, 
                                  nlp, find_span, sent_lemmatize, 
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
random_sentence_graph_file = 'data/extract_wiki/random_sentence_graph.pickle'

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
score_threshold = 0.6
max_sentence_length = 50
min_sentence_length = 5

# Some task specific classes

# Some helper functions
def collect_wiki_entity(file:str):
    return ['%s\t%s' % (line[9:re.search(r'" url="', line).start()], line[re.search(r'title="', line).end():re.search(r'">', line).start()]) for line in open(file).readlines() if re.match(r'^(<doc id=")', line) and line.isascii()]


def gen_kw_from_wiki_ent(wiki_ent:str, lower:bool=True):
    '''
    Generate entity name from the Wikipedia page title.
    Content in the bracket or after the first comma will be removed.
    ## Parameters
        wiki_ent: str
            The wikipedia page title or entity in the hyperlink
        lower: bool
            Whether the returned entity name should be lower-cased, default True
    '''
    if lower:
        wiki_ent = wiki_ent.lower()
    bracket_removed = remove_brackets(wiki_ent).strip()
    if bracket_removed:
        return ' '.join(word_tokenize(bracket_removed.split(',')[0]))
    else:
        return ' '.join(word_tokenize(wiki_ent))


def get_sentence(wiki_file:str, save_sent_file:str, save_cooccur_file:str, save_title_file:str):
    '''
    Collect sentences and entities from wikipedia dump file.
    ## Parameters
        wiki_file: str
            The file name of the wikipedia dump file
        save_sent_file: str
            The file where the collected sentences will be saved
        save_cooccur_file: str
            The file where the entity cooccurrence for each sentence will be saved
        save_title_file: str
            The file where the title for each sentence will be saved
    '''
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
                # Skip empty lines
                continue
            if re.match(r'^(<doc id=")', line):
                # This is the title line of a page, which is the first line of this page
                
                # Extract page title, extract entity name, insert
                page_name = ' '.join(line[re.search(r'title="', line).end():re.search(r'">', line).start()].split())
                page_kw = gen_kw_from_wiki_ent(page_name, True)
                wordtree, token2idx = build_word_tree_v2([page_kw])
                kw2ent_map = {page_kw : page_name}
            else:
                # This is a paragraph in the page

                kw2ent_map[page_kw] = page_name                         # Reload the entity for page title
                links = re.findall(r'<a href="[^"]*">[^<]*</a>', line)  # Extract all the links in this paragraph
                new_kws = []
                for l in links:
                    breakpoint = l.index('">')
                    entity_name = ' '.join(unquote(l[9:breakpoint]).split())    # Extract entity from the link
                    kw = gen_kw_from_wiki_ent(entity_name, False)               # Generate entity name for the entity, keep the original case
                    kw_lower = kw.lower()
                    # Update the mention-entity dict and the new entity list with lower-cased entity name
                    if kw == '':
                        print(wiki_file, line, entity_name)
                    else:
                        kw2ent_map[kw_lower] = entity_name
                        new_kws.append(kw_lower)
                    # Replace link with plain text
                    kw_in_text:str = l[breakpoint+2:-4]
                    kw_in_text_lower = kw_in_text.lower()
                    if kw_lower[-len(kw_in_text):] == kw_in_text_lower:         # Sometimes the entity name in the link is only part of the 
                        if kw_in_text.islower():                                # entity name we generate from the entity. We replace the 
                            line = line.replace(l, kw_lower)                    # original entity name with the name we create in some cases.
                        else:
                            line = line.replace(l, kw)
                    else:
                        line = line.replace(l, kw_in_text)
                paragraph = sent_tokenize(line)                                 # Split the paragraph into sentences
                wordtree, token2idx = build_word_tree_v2(new_kws, old_MyTree=wordtree, old_token2idx=token2idx) # Update the word tree with the new entity names from this paragraph
                co = CoOccurrence(wordtree, token2idx)
                for sent in paragraph:
                    sent = remove_brackets(sent)                                # Remove the content wrapped in brackets
                    reformed_sent = word_tokenize(sent)                         # Tokenize the sentence
                    reformed_sent = sent_lemmatize(sent)                        # Lemmatize the sentence
                    reformed_sent = [text.lower() for text in reformed_sent]    # Lower-case the sentence for finding entity name occurrence
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
    '''
    Get the index of the last token of an entity name in the SpaCy document
    '''
    while doc[idx].dep_ == 'compound':
        idx = doc[idx].head.i
    return idx


def get_ahead(doc, idx):
    '''
    Get the index of the first token of an entity name in the SpaCy document
    '''
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
    '''
    Get the complete span of a phrase in the SpaCy document
    '''
    phrase_right_most = get_back(doc, phrase_span[-1].i)
    phrase_left_most = min(get_ahead(doc, phrase_right_most), get_ahead(doc, phrase_span[0].i), phrase_span[0].i)
    return (phrase_left_most, phrase_right_most)

def cal_kw_recall(kw:str, full_phrase:str):
    return np.log(kw.count(' ') + 1 + const_num) / np.log(full_phrase.count(' ') + 1 + const_num)

def generate_clean_phrase(phrase:str):
    return ' '.join(re.sub(r'[^a-z0-9\s]', '', phrase).split())


def find_dependency_info_from_tree(doc, kw1, kw2):
    '''
    Find the dependency path that connect two entity names in the SpaCy document

    ## Return
        The steps from entity one to entity two or to the sub-root node
        The steps from entity two to entity two or to the sub-root node
        The corepath connecting the two entities
    '''
    # Find roots of the spans
    idx1 = find_root_in_span(kw1)
    idx2 = find_root_in_span(kw2)
    kw1_front, kw1_end = kw1[0].i, kw1[-1].i
    kw2_front, kw2_end = kw2[0].i, kw2[-1].i
    branch = np.zeros(len(doc))
    kw1_steps = []
    kw2_steps = []
    path_found = False
    
    # Start from entity one
    i = idx1
    while branch[i] == 0:
        branch[i] = 1
        kw1_steps.append(i)
        i = doc[i].head.i
        if i >= kw2_front and i <= kw2_end:
            # entity two is the parent of entity one
            path_found = True
            break
        
    if not path_found:
        # If entity two is not the parent of entity one, we start from entity two
        i = idx2
        while branch[i] != 1:
            branch[i] = 2
            kw2_steps.append(i)
            if i == doc[i].head.i:
                # If we reach the root of the tree, which hasn't been visited by the path from entity one, 
                # it means entity one and two are not in the same tree, no path is found
                return [], [], np.array([])
            
            i = doc[i].head.i
            if i >= kw1_front and i <= kw1_end:
                # entity one is the parent of entity two
                branch[branch != 2] = 0
                kw1_steps = []
                path_found = True
                break
    
    if not path_found:
        # entity one and entity two are on two sides, i is their joint
        break_point = kw1_steps.index(i)
        branch[kw1_steps[break_point+1 : ]] = 0
        kw1_steps = kw1_steps[:break_point] # Note that we remain the joint node in the branch, but we don't include joint point in kw1_steps and kw2_steps
                                            # this is because the joint node is part of the path and we need the modification information from it, 
                                            # but we don't care about its dependency
    branch[branch != 0] = 1             # Unify the branch to contain only 0s and 1s
    branch[kw1_front : kw1_end+1] = 1   # Mark the entity one as part of the branch
    branch[kw2_front : kw2_end+1] = 1   # Mark the entity two as part of the branch
    return kw1_steps, kw2_steps, branch


def get_path(doc, kw1_steps:List[int], kw2_steps:List[int]):
    '''
    Collect the corepath in str
    '''
    path_tokens = []
    for step in kw1_steps:
        path_tokens.append('i_' + doc[step].dep_)
    kw2_steps.reverse()
    for step in kw2_steps:
        path_tokens.append(doc[step].dep_)
    return ' '.join(path_tokens)


def reverse_path(path:str):
    path = path.split()
    r_path = ' '.join(['i_' + token if token[:2] != 'i_' else token[2:] for token in reversed(path)])
    return r_path


def gen_corepath_pattern(path:str):
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


def gen_subpath_pattern(path:str):
    return ' '.join(path.replace('compound', '').replace('conj', '').replace('appos', '').split())


def sentence_decompose(doc, kw1:str, kw2:str):
    '''
    Analyze the sentence with two entity names

    ## Return
    List of tuples. The returned tuples satisfy that:
        1. There exists a corepath starting with 'i_nsubj"
        2. The entity names are the complete span itself
    Each tuple contains the following fields: 
        span of entity one
        span of entity two
        a numpy array indicate the corepath
        the corepath in str
        the pattern generated from corepath
    '''
    kw1_spans = find_span(doc, kw1, True, True)
    kw2_spans = find_span(doc, kw2, True, True)
    data = []
    # A sentence may contain more than one occurrence for each entity name, we process each pair separately
    for kw1_span in kw1_spans:
        for kw2_span in kw2_spans:
            kw1_left_most, kw1_right_most = get_phrase_full_span(doc, kw1_span)
            kw2_left_most, kw2_right_most = get_phrase_full_span(doc, kw2_span)
            if kw1_left_most != kw1_span[0].i or kw1_right_most != kw1_span[-1].i or kw2_left_most != kw2_span[0].i or kw2_right_most != kw2_span[-1].i:
                # full span and keyword span don't match
                continue
            kw1_steps, kw2_steps, branch = find_dependency_info_from_tree(doc, kw1_span, kw2_span)
            if not branch.any():
                # If the branch is empty, it means no corepath is found
                continue
            path = get_path(doc, kw1_steps, kw2_steps)
            pattern = gen_corepath_pattern(path)
            if not pattern.startswith('i_nsubj'):
                # If the corepath does not start with 'i_nsubj', we drop it
                continue
            data.append((kw1_span, kw2_span, branch, path, pattern))
    return data


def load_pattern_freq(path_pattern_count_file_:str):
    c:Counter = my_read_pickle(path_pattern_count_file_)
    max_cnt = c.most_common(1)[0][1]
    log_max_cnt = np.log(max_cnt+1)
    return c, log_max_cnt


class CalFreq:
    def __init__(self, path_freq_file:str):
        c, log_max_cnt = load_pattern_freq(path_freq_file)
        self.c:Counter = c
        self.log_max_cnt:float = log_max_cnt

    def cal_freq_from_path(self, path:str):
        cnt = self.c.get(path)
        cnt = (cnt if cnt else 0.5) + 1
        return np.log(cnt) / self.log_max_cnt
    

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


modifier_dependencies = {'acl', 'advcl', 'advmod', 'amod', 'det', 'mark', 'meta', 'neg', 'nn', 'nmod', 'npmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl',
                         'appos', 'aux', 'auxpass', 'compound', 'cop', 'ccomp', 'xcomp', 'expl', 'punct', 'nsubj', 'csubj', 'csubjpass', 'dobj', 'iobj', 'obj', 'pobj'}
                
                
class FeatureProcess:
    '''
    Class that extracts features for scoring
    '''
    def __init__(self, sub_path_pattern_file:str):
        self.cal_freq = CalFreq(sub_path_pattern_file)
    
    def expand_dependency_info_from_tree(self, doc, branch:np.ndarray):
        dep_path:list = (np.arange(*branch.shape)[branch!=0]).tolist()
        for element in dep_path:
            if doc[element].dep_ == 'conj':
                branch[doc[element].head.i] = 0
        paths = collect_sub_dependency_path(doc, branch)
        paths = [item for item in paths if item[1].split()[0] in modifier_dependencies]
        for p in paths:
            pattern = gen_subpath_pattern(p[1])
            if pattern == '':
                branch[p[2]] = branch[p[0]]
            else:
                branch[p[2]] = self.cal_freq.cal_freq_from_path(pattern)
            
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
        '''
        Process one sentence which may contain several pairs of entities

        ## Parameters
            sent: str
                The sentence to be processed
            pairs: List of dict
                Information about the pairs. Each pair item should contain:
                    1. 'kw1' : entity name
                    2. 'kw2' : entity name
        
        ## Return:
            List of dict. The list will be empty if the length of the sentence is out of bound or no valid pair is found.
            Each item contains:
                Information about the valid entity pair. Each item contains:
                    1. 'kw1' : entity name
                    2. 'kw2' : entity name
                    3. 'kw1_span' : span for entity one
                    4. 'kw2_span' : span for entity two
                    5. 'pattern' : corepath pattern
                    6. 'path' : corepath
                    7. 'dep_coverage' : significance score
        '''
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


def informativeness_demo(sent:str, kw1:str, kw2:str, fp:FeatureProcess):
    doc = nlp(sent)
    kw1_span = find_span(doc, kw1, True, True)[0]
    kw2_span = find_span(doc, kw2, True, True)[0]
    kw1_steps, kw2_steps, branch = find_dependency_info_from_tree(doc, kw1_span, kw2_span)
    fp.expand_dependency_info_from_tree(doc, branch)
    context = []
    temp = []
    for i, checked in enumerate(branch):
        if checked:
            temp.append(doc[i].text)
        else:
            if temp:
                context.append(' '.join(temp))
                temp = []
    if temp:
        context.append(' '.join(temp))
    return pd.DataFrame({i:[doc[i].text, np.round(branch[i], 3)] for i in range(len(doc))})


class CollectSubPath:
    def __init__(self, similar_threshold:float):
        self.similar_threshold = similar_threshold
        
    def collect_subpath_pattern(self, doc, kw1:str, kw2:str)->List[str]:
        '''
        Collect subpath pattern between two entities from a SpaCy document

        ## Return
            List of subpath patterns collected from this document between the two entities
        '''
        data = []
        for kw1_span, kw2_span, branch, path, pattern in sentence_decompose(doc, kw1, kw2):
            ans = [str(item[1]) for item in collect_sub_dependency_path(doc, branch)]
            ans = [gen_subpath_pattern(item) for item in ans if 'punct' not in item]
            ans = [item for item in ans if item]
            data.extend(ans)
        return data


    def batched_collect_subpath_pattern(self, sent, pairs) -> List[str]:
        '''
        Collect subpath patterns from one sentence which may contain several pairs of entities

        ## Parameters
            sent: str
                The sentence to be processed
            pairs: List of dict
                Information about the pairs. Each pair item should contain:
                    1. 'kw1' : entity name
                    2. 'kw2' : entity name
        
        ## Return:
            List of subpath patterns. The list will be empty if the length of the sentence is out of bound or no valid pair is found.
        '''
        data = []
        pairs = [item for item in pairs if item[sim_str] >= self.similar_threshold]
        if not pairs:
            return []
        doc = nlp(sent)
        if len(doc) > max_sentence_length or len(doc) < min_sentence_length:
            return []
        for item in pairs:
            # Calculate calculate dependency coverage
            temp_data = self.collect_subpath_pattern(doc, item[kw1_str], item[kw2_str])
            data.extend(temp_data)
        return data


def process_line(sent:str, tups:List[Tuple[float, str, str]], sent_note:str, processor):
    '''
    Process a sentence with a given processor function.
    '''
    pairs = [{kw1_str:gen_kw_from_wiki_ent(tup[1], False), kw2_str:gen_kw_from_wiki_ent(tup[2], False), sim_str:tup[0], sent_str:sent_note, kw1_ent_str:tup[1], kw2_ent_str:tup[2]} for tup in tups]
    return processor(sent, pairs)
    
    
def process_list(sents:List[str], pairs_list:List[str], processor):
    '''
    Process a list of sentences with a given processor function.

    ## Parameters
        sents: list of str
            A list of sentences
        pairs_list: list of str
            Each str is a line in pair file
        processor: function
            A function to process the sentence
    '''
    data = []
    for line_idx, pairs in enumerate(tqdm.tqdm(pairs_list)):
        if not pairs:
            continue
        pairs = pairs.split('\t')
        tups = [eval(pair) for pair in pairs]
        data.extend(process_line(sents[line_idx], tups, sents[line_idx], processor))
    return data


def process_file(save_sent_file:str, save_pair_file:str, processor, posfix:str='.dat'):
    '''
    Process a file with a given processor function.
    '''
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


def filter_unrelated_from_df(df:pd.DataFrame, similar_threshold:float):
    return df[df[sim_str] >= similar_threshold]


def cal_freq_from_df(df:pd.DataFrame, cal_freq:CalFreq):
    return df.assign(pattern_freq = df.apply(lambda x: cal_freq.cal_freq_from_path(x[pattern_str]), axis=1))


def cal_score_from_df(df:pd.DataFrame):
    
    def cal_score(pattern_freq:float, dep_coverage:float):
        return 2 / ((1/pattern_freq)+(1/dep_coverage))

    sub_df = df.assign(score = df.apply(lambda x: cal_score(x[pattern_freq_str], x[dep_coverage_str]), axis=1))
    return sub_df


def get_entity_page(ent:str):
    '''
    Locate the wikipedia page of an entity in the dump
    '''
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
    '''
    Find all the paths connecting two entities in the graph.
    '''
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


def generate_sent_graph_from_graph(pairs:list, graph:nx.Graph, score_threshold:float, feature:str='score'):
    sent_graph = nx.Graph()
    for pair in tqdm.tqdm(pairs):
        data = graph.get_edge_data(*pair)
        sim = data['sim']
        data = data['data']
        new_data = []
        for score, note, span, dep_coverage, pattern_freq in data:
            if feature == 'explicit':
                feature_score = pattern_freq
            elif feature == 'significant':
                feature_score = dep_coverage
            else:
                feature_score = score
            if feature_score >= score_threshold:
                new_data.append((feature_score, {'score':score, 'note':note, 'span':span, 'significant':dep_coverage, 'explicit':pattern_freq}))
        if not new_data:
            continue
        new_data.sort(key=lambda x: x[0], reverse=True)
        new_data = list(zip(*new_data))[1]
        sent_graph.add_edge(*pair, sim=sim, data=new_data)
    return sent_graph
    
    
def generate_sent_graph_from_cooccur(pairs:list, cooccur:dict):
    sent_graph = nx.Graph()
    for ent1, ent2 in tqdm.tqdm(pairs):
        sent_candidates = list(cooccur[ent1] & cooccur[ent2])
        if len(sent_candidates) > 2:
            sents = sent_candidates[:2]
        else:
            sents = sent_candidates
        if not sents:
            continue
        new_data = [{'score':0, 'note':note, 'span':0, 'significant':0, 'explicit':0} for note in sents]
        sent_graph.add_edge(ent1, ent2, sim=0, data=new_data)
    return sent_graph


def generate_sample(target_graph:nx.Graph, source_graph:nx.Graph, ent1:str, ent2:str, max_hop_num:int=2, path_num:int=5, feature:str='score', replaceable=False):
    '''
    Generate dataset sample for a pair of entities

    ## Parameters
        target_graph: nx.Graph
            This graph provides target sentences
        source_graph: nx.Graph
            This graph provides input sentences
        max_hop_num: int
            The maximum number of hops in the path, 1 means 2 hop path, 2 means 3 hop path
        path_num: int
            The number of paths to be collected
        feature: str or None
            The feature to sort the paths. If None, the random paths will be selected. 
            The feature could be "score", "significant" and "explicit"
        replaceable: bool, default False
            Whether the input sentence in the path can be replaced with another sentence if it equals the target sentence.
            If not replaceable, when this happens, the path will be abandoned
    '''
    target = target_graph.get_edge_data(ent1, ent2)
    if not target:
        return None
    hop_num = 1
    triples = []
    target_note = target['data'][0]['note']     # Get the target sentence note
    paths = []
    while hop_num<=max_hop_num:
        temp_paths = find_path_between_pair(source_graph, ent1, ent2, hop_num=hop_num)
        path_candidates = []
        # Collect data for each path
        for path in temp_paths:
            temp_sum = 0
            path_abandon = False
            temp_path = []
            for i in range(len(path)-1):
                data = source_graph.get_edge_data(path[i], path[i+1])
                sim = data['sim']
                data = data['data']
                item:dict = deepcopy(data[0])
                # Handle the edge if the sentence equals the target
                if item['note'] == target_note:
                    if not replaceable or len(data) <= 1:
                        path_abandon = True
                        break
                    else:
                        item = deepcopy(data[1])
                item.update({'e1' : path[i], 'e2' : path[i+1], 'sim' : sim})
                temp_path.append(item)
                if feature:
                    temp_sum += 1 / item[feature]
            if path_abandon:
                continue
            if feature:
                path_candidates.append({'score' : (len(path)-1) / temp_sum, 'path' : temp_path})
            else:
                path_candidates.append(temp_path)
        
        if path_candidates:
            if feature:
                path_candidates.sort(key=lambda x: x['score'], reverse=True)
                path_candidates = [item['path'] for item in path_candidates]
            paths.extend(path_candidates)
            
        hop_num += 1
        
    if len(paths) < path_num:
        return None
    
    if not feature:
        # If feature is None, shuffle the paths
        random.seed(0)
        random.shuffle(paths)
        
    for path in paths[:path_num]:
        for tri in path:
            tri['pid'] = len(triples)
        triples.append(path)
        
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
        
        
    elif sys.argv[1] == 'collect_subpath_pattern_freq':
        
        sents = []
        pairs_list = []
        for file_idx in range(50):
            with open(save_sent_files[file_idx]) as f_in:
                sents.extend(f_in.read().split('\n'))
            with open(save_pair_files[file_idx]) as f_in:
                pairs_list.extend(f_in.read().split('\n'))
        
        collect_subpath = CollectSubPath(similar_threshold)
        c = Counter(process_list(sents, pairs_list, collect_subpath.batched_collect_subpath_pattern))
        my_write_pickle(sub_path_pattern_count_file, c)
        
        
    elif sys.argv[1] == 'collect_pattern_freq':
        
        sents = []
        pairs_list = []
        for file_idx in range(50):
            with open(save_sent_files[file_idx]) as f_in:
                sents.extend(f_in.read().split('\n'))
            with open(save_pair_files[file_idx]) as f_in:
                pairs_list.extend(f_in.read().split('\n'))
        
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
            data = cal_freq_from_df(data, c, log_max_cnt)
            data = cal_score_from_df(data)
            data.to_csv(save_selected_file, columns=record_columns, sep='\t', index=False)

        input_list = [(save_sent_files[i], save_pair_files[i], save_selected_files[i]) for i in range(len(save_sent_files))]
        _ = p.run(collect_dataset, input_list)
        
        
    elif sys.argv[1] == 'generate_graph':
        
        graph = generate_graph(save_selected_files, similar_threshold)
        my_write_pickle(graph_file, graph)
        
    
    elif sys.argv[1] == 'generate_sent_graph':
        
        graph = my_read_pickle(graph_file)
        threshold = score_threshold
        feature = 'score'
        if len(sys.argv) == 4:
            threshold = float(sys.argv[2])
            feature = sys.argv[3]
        single_sent_g = generate_sent_graph_from_graph(list(graph.edges), graph, threshold, feature)
        if len(sys.argv) == 4:
            my_write_pickle('sentence_graph_%s_%.2f.pickle' % (feature, threshold), single_sent_g)
        else:
            my_write_pickle(single_sent_graph_file, single_sent_g)
            
            
    elif sys.argv[1] == 'generate_random_sent_graph':
        
        d = my_read_pickle(entity_occur_from_cooccur_file)
        target_graph:nx.Graph = my_read_pickle(single_sent_graph_file)
        random_sent_g = generate_sent_graph_from_cooccur(list(target_graph.edges), d)
        my_write_pickle(random_sentence_graph_file, random_sent_g)
            
        
    # elif sys.argv[1] == 'collect_score_function_eval_dataset':
        
    #     graph:nx.Graph = my_read_pickle(graph_file)
    #     entity_occur_from_cooccur = my_read_pickle(entity_occur_from_cooccur_file)
    #     test_pairs = []
    #     for pair in random.sample(graph.edges, 500):
    #         ent1, ent2 = pair
    #         occur1 = entity_occur_from_cooccur.get(ent1)
    #         occur2 = entity_occur_from_cooccur.get(ent2)
    #         if occur1 is None or occur2 is None:
    #             continue
    #         intersect = occur1 & occur2
    #         if len(intersect) < 5:
    #             continue
    #         selected_sent = random.choice(graph.get_edge_data(*pair)['data'])[1]
    #         # print(selected_sent)
    #         temp_intersect = intersect - {selected_sent}
    #         random_sent = random.sample(temp_intersect, 1)[0]
    #         notes = [random_sent, selected_sent]
    #         random.shuffle(notes)
    #         for note in notes:
    #             test_pairs.append({'entity 1' : ent1, 'entity 2' : ent2, 'sentence' : note2line(note).strip(), 'score' : 0})
    #         if len(test_pairs) >= 100:
    #             break
    #     test_data = pd.DataFrame(test_pairs)
    #     test_data.to_csv('test.tsv', sep='\t', index=False)
        
        
    # elif sys.argv[1] == 'recalculate_score':
        
    #     for file in tqdm.tqdm(save_selected_files):
    #         with open(file) as f_in:
    #             data = pd.read_csv(f_in, sep='\t')
    #             data = cal_score_from_df(data)
    #         data.to_csv(file, columns=record_columns, sep='\t', index=False)
            
    
    elif sys.argv[1] == 'collect_sample_from_single_sent_graph':
        
        context_sent_score_threshold = 0.6
        target_graph:nx.Graph = my_read_pickle(single_sent_graph_file)
        target_edges = list(target_graph.edges)
        source_sent_g = my_read_pickle(graph_file)
        source_sent_g = generate_sent_graph_from_graph(list(source_sent_g.edges), source_sent_g, context_sent_score_threshold)
        
        sample_num = -1
        samples = []
        print(len(target_edges))
        edge_count = 0
        for edge_idx, edge in enumerate(tqdm.tqdm(target_edges)):
            edge_count += 1
            sample = generate_sample(target_graph, source_sent_g, edge[0], edge[1])
            if sample:
                samples.append(sample)
            if (sample_num > 0) and (len(samples) >= sample_num):
                break
        print(len(samples) * 1.0 / edge_count)
    
        with open('dataset_level_temp.json', 'w') as f_out:
            json.dump(samples, f_out)
            print(len(samples))
            
            
    elif sys.argv[1] == 'collect_sample_with_random_sentence':
        
        dataset_splits = ['train', 'dev', 'test']
        target_graph:nx.Graph = my_read_pickle(single_sent_graph_file)
        source_sent_g = my_read_pickle(random_sentence_graph_file)
        
        for dataset_split in dataset_splits:
            original_file = 'MyFiD/data/' + dataset_split + '.json'
            with open(original_file) as f_in:
                samples = json.load(f_in)
                print(original_file)
                new_samples = []
                for sample in tqdm.tqdm(samples):
                    new_samples.append(generate_sample(target_graph, source_sent_g, sample['pair'][0], sample['pair'][1], feature=None, replaceable=True))
                with open('random_' + dataset_split + '.json', 'w') as f_out:
                    json.dump(new_samples, f_out)
                            
                    
    elif sys.argv[1] == 'collect_sample_from_dataset':
        
        feature = sys.argv[2]
        dataset_file = sys.argv[3]
        source_graph_file = sys.argv[4]
        with open(dataset_file) as f_in:
            samples = json.load(f_in)
        target_edges = [sample['pair'] for sample in samples]
        target_graph:nx.Graph = my_read_pickle(single_sent_graph_file)
        source_sent_g:nx.Graph = my_read_pickle(source_graph_file)
        
        sample_num = -1
        samples = []
        print(len(target_edges))
        edge_count = 0
        fail_num = 0
        for edge_idx, edge in enumerate(tqdm.tqdm(target_edges)):
            edge_count += 1
            sample = generate_sample(target_graph, source_sent_g, edge[0], edge[1], path_num=1, feature=feature)
            if sample:
                samples.append(sample)
            else:
                sample = generate_sample(target_graph, target_graph, edge[0], edge[1], path_num=1)
                if sample:
                    fail_num += 1
                    samples.append(sample)
                else:
                    print(edge)
            if (sample_num > 0) and (len(samples) >= sample_num):
                break
        print(len(samples) * 1.0 / edge_count)
        print('fail on', fail_num)
    
        with open('dataset_' + feature + '_temp.json', 'w') as f_out:
            json.dump(samples, f_out)
            print(len(samples))
            
    
    # elif sys.argv[1] == 'collect_second_level_sample':
        
    #     with open('dataset_level_1.json') as f_in:
    #         samples = json.load(f_in)
        
    #     second_level_samples = [generate_second_level_sample(sample) for sample in tqdm.tqdm(samples)]
    #     with open('dataset_level_2.json', 'w') as f_out:
    #         json.dump(second_level_samples, f_out)
    #         print(len(second_level_samples))
        
    # elif sys.argv[1] == 'collect_triangles_from_graph':
        
    #     with bz2.open(w2vec_dump_file) as f_in:
    #         w2vec = Wikipedia2Vec.load(f_in)
            
    #     graph = my_read_pickle(single_sent_graph_file)
    #     edges = [edge for edge in tqdm.tqdm(graph.edges) if graph.get_edge_data(*edge)['score'] > 0.65]
    #     filtered_graph = graph.edge_subgraph(edges)
    #     nodes = []
    #     for node in tqdm.tqdm(filtered_graph):
    #         ent = w2vec.get_entity(node)
    #         if ent is None:
    #             continue
    #         if ent.count >= 20:
    #             nodes.append(node)
    #     filtered_graph = filtered_graph.subgraph(nodes)
    #     triangle_set = find_all_triangles(filtered_graph)
    #     my_write_pickle('data/extract_wiki/triangles.pickle', triangle_set)