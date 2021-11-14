from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer
import datasets
import pandas as pd
from nltk.corpus import stopwords
import tqdm
from typing import List

# Some global variable
train_source = 'cnn_cln/train.source'
train_target = 'cnn_cln/train.target'
valid_source = 'cnn_cln/val.source'
valid_target = 'cnn_cln/val.target'
test_source = 'cnn_cln/test.source'
test_target = 'cnn_cln/test.target'
dataset_dir = 'cnn_summary'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
stop_words = set(stopwords.words('english'))

def remove_stopwords(sents:List[str]):
    return [' '.join([w for w in sent.split() if not w.lower() in stop_words]) for sent in tqdm.tqdm(sents)]

print('Generate dataset for training data')
with open(train_source) as f_in:
    sents = f_in.readlines()
    train_source_sents = remove_stopwords(sents)

print('Generate dataset for validation data')
with open(valid_source) as f_in:
    sents = f_in.readlines()
    valid_source_sents = remove_stopwords(sents)
    
print('Generate dataset for test data')
with open(test_source) as f_in:
    sents = f_in.readlines()
    test_source_sents = remove_stopwords(sents)
    
    
# [Build] huggingface dataset 
train_df = datasets.Dataset.from_pandas(pd.DataFrame({'source' : train_source_sents, 'summary' : open(train_target)}))
valid_df = datasets.Dataset.from_pandas(pd.DataFrame({'source' : valid_source_sents, 'summary' : open(valid_target)}))
test_df = datasets.Dataset.from_pandas(pd.DataFrame({'source' : test_source_sents, 'summary' : open(test_target)}))

ds = datasets.DatasetDict()
ds['train'] = train_df
ds['valid'] = valid_df
ds['test'] = test_df
ds.save_to_disk(dataset_dir)

def tokenize_function(examples):
    ret = tokenizer(examples['source'], padding='max_length', max_length=600, truncation=True)
    with tokenizer.as_target_tokenizer():
        ret['labels'] = tokenizer(examples['summary'], padding='max_length', max_length=150, truncation=True)['input_ids']
    return ret

tokenized_datasets = ds.map(tokenize_function, batched=True, batch_size=2)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

train_arg = TrainingArguments(dataset_dir)

trainer = Trainer(
    model=model, args=train_arg, train_dataset=tokenized_datasets['train'], eval_dataset=tokenized_datasets['valid']
)

trainer.train()

trainer.save_model(dataset_dir)