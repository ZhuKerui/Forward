from typing import List
from statistics import mean
import json
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file:str,
                 global_rank=-1, 
                 world_size=-1):
        self.load_data(data_file, global_rank=global_rank, world_size=world_size)
        
    def load_data(self, data_file:str, global_rank=-1, world_size=-1):
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

