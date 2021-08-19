from transformers import EncoderDecoderModel, BertTokenizer, BertModel, BertForNextSentencePrediction, BatchEncoding
import torch
from torch.nn import Softmax, Linear, CrossEntropyLoss
from torch.nn.functional import normalize, log_softmax
from collections import defaultdict
import sys

sys.path.append('..')
from tools.BasicUtils import *

# Shared variables
default_base_model = 'bert-base-uncased'

my_tokenizer = BertTokenizer.from_pretrained(default_base_model)
my_tokenizer.add_special_tokens({'additional_special_tokens' : ['<RELATION>']})
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_tokenize(sents:List[str], sents2:List[str]=None, device:torch.device=None):
    __device = device if device is not None else my_device
    if sents2 is None:
        return BatchEncoding(my_tokenizer(sents, padding=True, truncation=True, max_length=100, return_tensors="pt")).to(__device)
    else:
        return BatchEncoding(my_tokenizer(sents, sents2, padding=True, truncation=True, max_length=100, return_tensors="pt")).to(__device)

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
    def __init__(self, model_file:str=default_base_model, device:str=None):
        super().__init__()
        self._score_function = BertForNextSentencePrediction.from_pretrained(model_file)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._sm = Softmax(1)
        
        self._score_function.resize_token_embeddings(len(my_tokenizer))
        self._score_function.to(self._device)

    def forward(self, candidate_sents:List[str], query:str):
        inputs = my_tokenize(candidate_sents, [query]*len(candidate_sents), self._device)
        candidate_sents_ids = my_tokenize(candidate_sents, device=self._device)['input_ids']
        output = self._score_function(**inputs, labels=torch.LongTensor([1]*len(candidate_sents)).to(self._device), output_hidden_states=True)
        return self._sm(output.logits)[:, 1], candidate_sents_ids


class ScoreFunction2(torch.nn.Module):
    def __init__(self, context_model_file:str=default_base_model, query_model_file:str=default_base_model, device:str=None):
        super().__init__()
        self._context_encoder = BertModel.from_pretrained(context_model_file)
        self._query_encoder = BertModel.from_pretrained(query_model_file)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        self._query_encoder.resize_token_embeddings(len(my_tokenizer))
        self._query_encoder.to(self._device)
        self._context_encoder.to(self._device)

    def forward(self, candidate_sents:List[str], query:str):
        context_inputs = my_tokenize(candidate_sents, self._device)
        query_inputs = my_tokenize(query, self._device)
        context_emb = normalize(self._context_encoder(**context_inputs).last_hidden_state[:, 0, :])
        query_emb = normalize(self._query_encoder(**query_inputs).last_hidden_state[:, 0, :])
        return torch.inner(context_emb, query_emb).squeeze(-1), context_inputs['input_ids']


class DenseRetrieverWithScore(torch.nn.Module):
    def __init__(self, sparse_retriever:SparseRetrieveSentForPairCoOccur, score_function):
        super().__init__()
        self.sparse_retriever = sparse_retriever
        self.sf = score_function

    def forward(self, kw1_list:List[str], kw2_list:List[str], top_k:int=5):
        batches = []
        with torch.no_grad():
            for kw1, kw2 in zip(kw1_list, kw2_list):
                candidate_sents = self.sparse_retriever.retrieve(kw1, kw2)
                query = '%s <RELATION> %s' % (kw1, kw2)
                sf_output = [self.sf(sents, query) for sents in batch(candidate_sents, 16)]
                scores = torch.cat([score for score, ids in sf_output])
                top_idx = ntopidx(top_k, scores)
                temp_candidate_sents = [candidate_sents[idx] for idx in top_idx]
                batches.append((temp_candidate_sents, query))
                torch.cuda.empty_cache()

        scores = torch.cat([self.sf(sents, query).unsqueeze(0) for sents, query in batches])
        retrieved_sents = [item[0] for item in batches]
        return retrieved_sents, scores


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


class Reader3(torch.nn.Module):
    def __init__(self, encoder_model:str, decoder_model:str, tokenizer:BertTokenizer, device:str=None):
        super().__init__()
        self._encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._tokenizer = tokenizer
        self._encoder_decoder.encoder.resize_token_embeddings(len(self._tokenizer))
        self._encoder_decoder.decoder.resize_token_embeddings(len(self._tokenizer))

    def forward(self, sents:List[str], score:torch.Tensor, rel:List[str]=None):
        inputs = my_tokenize(sents, self._device)
        if rel is not None:
            with my_tokenizer.as_target_tokenizer():
                targets = my_tokenize(rel, device=self._device)
            gen_output = self._encoder_decoder(input_ids = inputs['input_ids'], labels = targets['input_ids'])
        sents_emb = self._encoder(**inputs).last_hidden_state[:, 0, :]
        merged_emb = torch.inner(score, sents_emb)
        cls_ret = self._classifier(merged_emb)
        if rel is not None:
            temp_cls = self._rel2cls[rel]
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)], self._loss_cal(cls_ret, torch.tensor(temp_cls, dtype=torch.long))
        else:
            return self._rels[torch.argmax(self._sm(cls_ret), dim=1)]


# Helper functions

def demo_score_function(kw1, kw2, sf, sparse_retriever:SparseRetrieveSentForPairCoOccur):
    candidate_sents = sparse_retriever.retrieve(kw1, kw2)
    with torch.no_grad():
        scores = torch.cat([sf(sents, '%s <RELATION> %s' % (kw1, kw2)) for sents in batch(candidate_sents, 16)])
    torch.cuda.empty_cache()
    return scores, candidate_sents

def demo_reader(kw1, kw2, reader, sparse_retriever:SparseRetrieveSentForPairCoOccur):
    candidate_sents = sparse_retriever.retrieve(kw1, kw2)
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rels = reader(candidate_sents[:5], torch.rand((1,5)).to(device))
        torch.cuda.empty_cache()
    return rels

def get_nll(
    seq_logits:torch.Tensor, doc_scores:torch.Tensor, target:torch.Tensor, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs:int=5
) -> torch.Tensor:
    # shift tokens left
    pad_token_id = my_tokenizer.pad_token_id
    target = torch.cat(
        [target[:, 1:], target.new(target.shape[0], 1).fill_(pad_token_id)], 1
    )

    # bos_token_id is None for T5
    bos_token_id = my_tokenizer.bos_token_id
    use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

    def _mask_pads(ll:torch.Tensor, smooth_obj:torch.Tensor):
        pad_mask = target.eq(pad_token_id)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
            smooth_obj.masked_fill_(pad_mask, 0.0)
        return ll.squeeze(-1), smooth_obj.squeeze(-1)

    # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
    seq_logprobs = log_softmax(seq_logits, dim=-1).view(
        seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
    )  # batch_size x n_docs x tgt_len x #vocab_size
    doc_logprobs = log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

    # RAG-sequence marginalization
    first_token_scores = seq_logprobs[:, :, :1, :]
    second_token_scores = seq_logprobs[:, :, 1:2, :]
    remainder = seq_logprobs[:, :, 2:, :]
    rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

    # calculate loss
    target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
    assert target.dim() == rag_logprobs.dim()

    ll = rag_logprobs.gather(dim=-1, index=target)
    smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

    ll, smooth_obj = _mask_pads(ll, smooth_obj)

    # sum over tokens, exclude bos while scoring
    ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
    smooth_obj = smooth_obj.sum(2)
    ll = ll.logsumexp(1)  # logsumexp over docs
    smooth_obj = smooth_obj.logsumexp(1)

    nll_loss = -ll
    smooth_loss = -smooth_obj

    if reduce_loss:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / rag_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss