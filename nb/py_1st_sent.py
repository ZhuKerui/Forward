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

if __name__ == '__main__':
    import pandas as pd
    from transformers import BertTokenizer, BertForSequenceClassification, BatchEncoding, AdamW
    import torch
    from typing import Iterable
    import tqdm

    # Load training and validation data
    train_df = pd.read_csv('train.csv')
    valid_df = pd.read_csv('valid.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained('temp2.pt')

    # Function for batch generation
    def batch(sents:Iterable, n:int):
        l = len(sents)
        for ndx in range(0, l, n):
            yield sents[ndx:min(ndx + n, l)]
    
    # Train the model
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)

    batch_list = [item for item in batch(train_df, 32)]

    for epoch in range(3):
        loss = 0
        batch_num = 0
        for batch_df in tqdm.tqdm(batch_list):
            optim.zero_grad()
            labels = torch.tensor([1 if i == 'T' else 0 for i in batch_df.label.to_list()]).unsqueeze(1).to(device)
            # inputs = BatchEncoding(tokenizer(batch_df.sent.to_list(), batch_df.head_ent.to_list(), padding=True, truncation=True, max_length=80, return_tensors='pt')).to(device)
            inputs = BatchEncoding(tokenizer(batch_df.sent.to_list(), batch_df.head_ent.to_list(), padding=True, truncation=True, max_length=80, return_tensors="pt")).to(device)
            # print(inputs)
            # break
            output = model(**inputs, labels=labels)
            loss += output.loss
            output.loss.backward()
            optim.step()
        print(loss / len(batch_list))

    # Save trained model
    model.save_pretrained('temp2.pt')