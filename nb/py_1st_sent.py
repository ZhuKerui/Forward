import re
from nltk import sent_tokenize
import sys

sys.path.append('..')
from tools.BasicUtils import get_wiki_page_from_kw

remove_list = ['See also', 'References', 'Further reading']

def collect_neg_sents_from_term(term:str, n:int=5):
    page = get_wiki_page_from_kw(term)
    if page is None:
        return None
    if page.content.lower().count(term) < (n * 2):
        return None
    neg_sents = []
    section_list = page.sections.copy()
    for item in remove_list:
        if item in section_list:
            section_list.remove(item)
    while len(neg_sents) < n and len(section_list) != 0:
        section = section_list.pop()
        section_text = page.section(section)
        if section_text is None:
            continue
        section_text = section_text.lower()
        if term not in section_text:
            continue
        # Remove {} and ()
        while re.search(r'{[^{}]*}', section_text):
            section_text = re.sub(r'{[^{}]*}', '', section_text)
        while re.search(r'\([^()]*\)', section_text):
            section_text = re.sub(r'\([^()]*\)', '', section_text)
        if term not in section_text:
            continue
        processed_text = ' '.join(section_text.split())
        temp_sents = sent_tokenize(processed_text)
        for sent in temp_sents:
            if term in sent:
                neg_sents.append('%s\t%s' % (term, re.sub(r'[^A-Za-z0-9,.\s-]', '', sent.strip())))
                if len(neg_sents) >= n:
                    break
    return '\n'.join(neg_sents) if neg_sents else None