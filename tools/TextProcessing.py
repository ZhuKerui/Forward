import json
from tools.BasicUtils import my_read, my_write
import numpy as np
import re
from typing import Dict
from nltk import WordNetLemmatizer, pos_tag, word_tokenize, sent_tokenize
import spacy
nlp = spacy.load('en_core_web_sm')

wnl = WordNetLemmatizer()

def process_keywords(keywords:list):
    filtered_words = set(['can', 'it', 'work', 'in', 'parts', 'is', 'its', 'or', 'and', 'a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ''])

    stable_kw = []
    unstable_kw = []
    for kw in keywords:
        if '- ' in kw:
            continue
        splited = kw.replace('-', ' - ')
        reformed = ' '.join(sent_lemmatize(splited))
        if reformed in filtered_words:
            continue
        if reformed == splited:
            stable_kw.append(kw)
        else:
            unstable_kw.append('%s\t%s' % (kw, reformed))
    return stable_kw, unstable_kw
    
def clean_text(text:str):
    while re.search(r'{[^{}]*}', text):
        text = re.sub(r'{[^{}]*}', '', text)
    while re.search(r'\([^()]*\)', text):
        text = re.sub(r'\([^()]*\)', '', text)
    while re.search(r'\[[^][]*\]', text):
        text = re.sub(r'\[[^][]*\]', '', text)
    return ' '.join(re.sub(r'[^a-z0-9,.;\s-]', '', text.lower()).replace('-', ' - ').strip().split())

def build_word_tree(input_txt:str, dump_file:str, entity_file:str):
    MyTree = {}
    entities = []
    cnt = 0
    with open(input_txt, 'r', encoding='utf-8') as load_file:
        for word in load_file:
            # Directly add the '_' connected keyword
            word = word.strip()
            phrase = word.replace('-', ' - ').split()
            if not phrase:
                print(word)
                continue
            entities.append('_'.join(phrase))
            cnt += 1
            # Insert the keyword to the tree structure
            if len(phrase) == 1:
                # If the word is an atomic word instead of a phrase
                if word not in MyTree.keys():
                    # If this is the first time that this word is inserted to the tree
                    MyTree[word] = {"":""}
                elif "" not in MyTree[word].keys():
                    # If the word has been inserted but is viewed as an atomic word the first time
                    MyTree[word][""] = ""
                # If the word has already been inserted as an atomic word, then we do nothing
            else:
                # If the word is an phrase
                length = len(phrase)
                fw = phrase[0]
                if fw not in MyTree.keys():
                    MyTree[fw] = {}
                temp_dict = MyTree.copy()
                parent_node = fw
                for i in range(1, length):
                    if phrase[i]:
                        sw = phrase[i]
                        if sw not in temp_dict[parent_node].keys():
                            # The second word is inserted to as the child of parent node the first time
                            temp_dict[parent_node][sw] = {}
                        if i == length - 1:
                            # If the second word is the last word in the phrase
                            if "" not in temp_dict[parent_node][sw].keys():
                                temp_dict[parent_node][sw][""] = ""
                        else:
                            # If the second word is not the last word in the phrase
                            temp_dict = temp_dict[parent_node].copy()
                            parent_node = sw
        print('Building word tree is accomplished with %d words added' % (cnt))
    with open(dump_file, 'w', encoding='utf-8') as output_file:
        json.dump(MyTree, output_file)
    my_write(entity_file, entities)
        
def sent_lemmatize(sentence:str):
    return [str(wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word) for word, tag in pos_tag(word_tokenize(sentence))]

def batched_sent_tokenize(paragraphs:list):
    content = []
    for para in paragraphs:
        content += sent_tokenize(para)
    return content

def find_dependency_path(sent:str, kw1:str, kw2:str):
    doc = nlp(sent)
    kw1_spans = find_span(doc, kw1, True)
    kw2_spans = find_span(doc, kw2, True)
    if len(kw1_spans) != 1 or len(kw2_spans) != 1:
        return ''
    return find_dependency_path_from_tree(doc, doc[kw1_spans[0][0]:kw1_spans[0][1]], doc[kw2_spans[0][0]:kw2_spans[0][1]])

def find_dependency_path_from_tree(doc, kw1:spacy.tokens.span.Span, kw2:spacy.tokens.span.Span):
    idx1 = kw1[-1].i
    idx2 = kw2[-1].i
    branch = np.zeros(len(doc))
    i = idx1
    while branch[i] == 0:
        branch[i] = 1
        i = doc[i].head.i
    i = idx2
    while branch[i] != 1:
        branch[i] = 2
        if i == doc[i].head.i:
            return ''
        i = doc[i].head.i
    dep1 = []
    j = idx1
    while j != i:
        dep1.append('i_%s' % doc[j].dep_)
        j = doc[j].head.i
    dep2 = []
    j = idx2
    while j != i:
        dep2.append(doc[j].dep_)
        j = doc[j].head.i
    dep2.reverse()
    if branch[idx2] == 1:
        # kw2 is along the heads of kw1
        return ' '.join(dep1)
    elif i == idx1:
        # kw1 is along the heads of kw2
        return ' '.join(dep2)
    else:
        return ' '.join(dep1 + dep2)

def find_span(doc:spacy.tokens.doc.Doc, phrase:str, use_lemma:bool=False):
    phrase_tokens = phrase.split()
    phrase_length = len(phrase_tokens)
    sent_tokens = [str(t.lemma_ if use_lemma else t) for t in doc]
    return [(i, i + phrase_length) for i in range(len(doc)-phrase_length+1) if phrase_tokens == sent_tokens[i : i + phrase_length]]