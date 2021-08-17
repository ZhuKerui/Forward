from transformers import EncoderDecoderModel, BertTokenizer, BertModel, BertForNextSentencePrediction, BatchEncoding
import torch
from torch.nn import Softmax, Linear, CrossEntropyLoss
from torch.nn.functional import normalize
from collections import defaultdict
import sys

sys.path.append('..')
from tools.BasicUtils import *

# Models

class SparseRetrieveSentForPairCoOccur:
    def __init__(self, sent_file:str, occur_file:str):
        self._sents = my_read(sent_file)
        self._occur_dict = defaultdict(set)
        for k, v in json.load(open(occur_file)).items():
            self._occur_dict[k] = set(v)

    def retrieve(self, kw1:str, kw2:str):
        co_occur_index = self._occur_dict[kw1] & self._occur_dict[kw2]
        return [self._sents[idx] for idx in co_occur_index]


class ScoreFunction1(torch.nn.Module):
    def __init__(self, model_file:str, additional_special_tokens:List[str]=None, device:str=None):
        super().__init__()
        self._score_function = BertForNextSentencePrediction.from_pretrained(model_file)
        self._tokenizer = BertTokenizer.from_pretrained(model_file)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._sm = Softmax(1)
        
        if additional_special_tokens is not None:
            self._tokenizer.add_special_tokens({'additional_special_tokens' : additional_special_tokens})
            self._score_function.resize_token_embeddings(len(self._tokenizer))
        self._score_function.to(self._device)

    def forward(self, candidate_sents:List[str], query:str):
        inputs = BatchEncoding(self._tokenizer(candidate_sents, [query]*len(candidate_sents), padding=True, truncation=True, max_length=80, return_tensors="pt")).to(self._device)
        output = self._score_function(**inputs, labels=torch.LongTensor([1]*len(candidate_sents)).to(self._device), output_hidden_states=True)
        return self._sm(output.logits)[:, 1]


class ScoreFunction2(torch.nn.Module):
    def __init__(self, context_model_file:str, query_model_file:str, additional_special_tokens:List[str]=None, device:str=None):
        super().__init__()
        self._context_encoder = BertModel.from_pretrained(context_model_file)
        self._query_encoder = BertModel.from_pretrained(query_model_file)
        self._tokenizer = BertTokenizer.from_pretrained(query_model_file)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        if additional_special_tokens is not None:
            self._tokenizer.add_special_tokens({'additional_special_tokens' : additional_special_tokens})
            self._query_encoder.resize_token_embeddings(len(self._tokenizer))
        self._query_encoder.to(self._device)
        self._context_encoder.to(self._device)

    def forward(self, candidate_sents:List[str], query:str):
        context_inputs = BatchEncoding(self._tokenizer(candidate_sents, padding=True, truncation=True, max_length=80, return_tensors="pt")).to(self._device)
        query_inputs = BatchEncoding(self._tokenizer(query, padding=True, truncation=True, max_length=20, return_tensors="pt")).to(self._device)
        context_emb = normalize(self._context_encoder(**context_inputs).last_hidden_state[:, 0, :])
        query_emb = normalize(self._query_encoder(**query_inputs).last_hidden_state[:, 0, :])
        return torch.inner(context_emb, query_emb).squeeze(-1)


class Reader1(torch.nn.Module):
    def __init__(self, encoder_model:str, rels:List[str], device:str=None):
        super().__init__()
        self._rel2cls = {rel:i for i, rel in enumerate(rels)}
        self._rels = rels
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._classifier = Linear(768, len(self._rel2cls), device=self._device)
        self._encoder = BertModel.from_pretrained(encoder_model).to(self._device)
        self._tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self._loss_cal = CrossEntropyLoss()
        self._sm = Softmax(1)

    def forward(self, sents:List[str], score:torch.Tensor, rel:str=None):
        inputs = BatchEncoding(self._tokenizer(sents, padding=True, truncation=True, max_length=80, return_tensors="pt")).to(self._device)
        sents_emb = self._encoder(**inputs).last_hidden_state[:, 0, :]
        merged_emb = normalize(torch.matmul(score, sents_emb))
        cls_ret = self._classifier(merged_emb)
        if rel is not None:
            temp_cls = self._rel2cls[rel]
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)], self._loss_cal(cls_ret, torch.tensor(temp_cls, dtype=torch.long).to(self._device))
        else:
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)]


class Reader2(torch.nn.Module):
    def __init__(self, encoder_model:str, rels:List[str], device:str=None):
        super().__init__()
        self._rel2cls = {rel:i for i, rel in enumerate(rels)}
        self._rels = rels
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._classifier = Linear(768, len(self._rel2cls), device=self._device)
        self._encoder = BertModel.from_pretrained(encoder_model).to(self._device)
        self._tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self._loss_cal = CrossEntropyLoss()
        self._sm = Softmax(1)

    def forward(self, sents:List[str], score:torch.Tensor, rel:str=None):
        inputs = BatchEncoding(self._tokenizer(sents, padding=True, truncation=True, max_length=80, return_tensors="pt")).to(self._device)
        sents_emb = self._encoder(**inputs).last_hidden_state[:, 0, :]
        merged_emb = torch.inner(score, sents_emb)
        cls_ret = self._classifier(merged_emb)
        if rel is not None:
            temp_cls = self._rel2cls[rel]
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)], self._loss_cal(cls_ret, torch.tensor(temp_cls, dtype=torch.long))
        else:
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)]


            
# Helper functions

def demo_score_function(kw1, kw2, sf, retriever:SparseRetrieveSentForPairCoOccur):
    candidate_sents = retriever.retrieve(kw1, kw2)
    with torch.no_grad():
        scores = torch.cat([sf(sents, '%s <RELATION> %s' % (kw1, kw2)) for sents in batch(candidate_sents, 16)])
    torch.cuda.empty_cache()
    return scores, candidate_sents

def demo_reader(kw1, kw2, reader, retriever:SparseRetrieveSentForPairCoOccur):
    candidate_sents = retriever.retrieve(kw1, kw2)
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rels = reader(candidate_sents[:5], torch.rand((1,5)).to(device))
        torch.cuda.empty_cache()
    return rels