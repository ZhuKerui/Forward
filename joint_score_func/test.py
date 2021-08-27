import json
import csv
from typing import List
import numpy as np
import torch
from torch.nn import Softmax, Sigmoid, BCELoss
import pandas
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BatchEncoding, PreTrainedModel, PretrainedConfig, BertModel, BertConfig
import sys

sys.path.append('..')
from joint_score_func import SparseRetrieveSentForPairCoOccur
import pdb

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({'additional_special_tokens' : ['<RELATION>']})

dataset = load_from_disk('data/single-ollie2')
training_args = TrainingArguments("data/single-ollie2", evaluation_strategy="epoch")

class ScoreFunction2(PreTrainedModel):
    def __init__(self, config:PretrainedConfig):
        super().__init__(config)
        self._context_encoder = BertModel(config)
        self._query_encoder = BertModel(config)
        self._sigmoid = Sigmoid()

    def forward(self, 
        context_input_ids,
        query_input_ids,
        context_token_type_ids=None,
        context_attention_mask=None,
        query_token_type_ids=None,
        query_attention_mask=None):
        context_inputs = {'input_ids': context_input_ids, 'token_type_ids': context_token_type_ids, 'attention_mask': context_attention_mask}
        query_inputs = {'input_ids': query_input_ids, 'token_type_ids': query_token_type_ids, 'attention_mask': query_attention_mask}
        context_emb = self._context_encoder(**context_inputs).last_hidden_state[:, 0, :]
        query_emb = self._query_encoder(**query_inputs).last_hidden_state[:, 0, :]
        score = self._sigmoid(torch.mul(context_emb, query_emb).sum(dim=1))
        return score

class ScoreFunction2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_function = BCELoss()
        loss = loss_function(outputs, labels)
        return (loss, outputs) if return_outputs else loss

def preprocess_sf2(examples):
    query = ['%s <RELATION> %s' % (ent1, ent2) for ent1, ent2 in zip(examples['ent1'], examples['ent2'])]
    context_tokenized = tokenizer(examples["sent"], padding=True, truncation=True, max_length=100)
    query_tokenized = tokenizer(query, padding=True, truncation=True, max_length=100)
    return {'context_input_ids': context_tokenized['input_ids'], 
            'context_token_type_ids': context_tokenized['token_type_ids'], 
            'context_attention_mask': context_tokenized['attention_mask'],
            'query_input_ids': query_tokenized['input_ids'], 
            'query_token_type_ids': query_tokenized['token_type_ids'], 
            'query_attention_mask': query_tokenized['attention_mask']}

train_dataset = dataset['train'].map(preprocess_sf2, batched=True)
valid_dataset = dataset['valid'].map(preprocess_sf2, batched=True)

model = ScoreFunction2(BertConfig())
model._query_encoder.resize_token_embeddings(len(tokenizer))

trainer = ScoreFunction2Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)

pdb.set_trace()
trainer.train()