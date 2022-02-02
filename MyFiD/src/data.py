from typing import List
from statistics import mean
from copy import deepcopy
import json
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file:str,
                 n_context:int,
                 global_rank=-1, 
                 world_size=-1,
                 duplicate_sample=True):
        self.n_context = n_context
        self.load_data(data_file, global_rank=global_rank, world_size=world_size, duplicate_sample=duplicate_sample)
        
    def load_data(self, data_file:str, global_rank=-1, world_size=-1, duplicate_sample=True):
        with open(data_file) as f_in:
            samples = json.load(f_in)
        
        data = []
        for k, sample in enumerate(samples):
            if global_rank > -1 and not k%world_size==global_rank:
                continue
            answer:str = sample['target']
            sources:List[str] = sample['source']
            entity:List[str] = sample['entity']
            triple:list = sample['triple']
            avg_scores = [mean([tri['score'] for tri in path]) for path in triple]
            sorted_list = sorted(zip(avg_scores, triple), key=lambda x: x[0], reverse=True)
            if duplicate_sample:
                while len(sorted_list) < self.n_context:
                    append_list = deepcopy(sorted_list[:self.n_context - len(sorted_list)])
                    sorted_list.extend(append_list)
            sorted_list = sorted_list[:self.n_context]
            triple = list(zip(*sorted_list))[1]
            contexts = [[{'e1' : entity[tri['e1']], 
                          'e2' : entity[tri['e2']], 
                          'sent' : sources[tri['sent']],
                          'score' : tri['score']} for tri in path] for path in triple]
            
            data.append({'target_pair' : sample['pair'],
                         'target' : answer,
                         'ctxs' : contexts,
                         'index' : k})
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    
class FiDDataset(Dataset):
    def __init__(self, data_file: str, n_context: int, global_rank=-1, world_size=-1, no_sent=False,
                 no_path=False, duplicate_sample=True):
        super().__init__(data_file, n_context, global_rank, world_size)
        self.no_sent = no_sent
        self.no_path = no_path
        
    def __getitem__(self, index:int):
        example = self.data[index]
        question = 'entity1: {} entity2: {}'.format(*example['target_pair']).lower()
        target = example['target']
        contexts = []
        for ctx in example['ctxs']:
            path = [ctx[0]['e1']]
            sents = []
            for i, tri in enumerate(ctx):
                path.append(tri['e2'])
                sents.append('sentence%d: %s' % (i+1, tri['sent']))
            path = '; '.join(path)
            sents = ' '.join(sents)
            contexts.append('%s %s %s' % (question, 'path: ' + path if not self.no_path else '', sents if not self.no_sent else ''))

        return {
            'index' : index,
            'target' : target,
            'passages' : contexts
        }


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='longest',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids:torch.Tensor = target["input_ids"]
        target_mask:torch.Tensor = target["attention_mask"]
        target_mask = target_mask.bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        text_passages = [example['passages'] for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)
    