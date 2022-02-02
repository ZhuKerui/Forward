from copy import deepcopy
from src.data import Dataset
import torch


class FiDDataset(Dataset):
    def __init__(self, data_file: str, n_context: int, global_rank=-1, world_size=-1, no_sent=False,
                 no_path=False, duplicate_sample=True):
        super().__init__(data_file, global_rank, world_size)
        self.n_context = n_context
        self.no_sent = no_sent
        self.no_path = no_path
        self.duplicate_sample = duplicate_sample
        
    def __getitem__(self, index:int):
        example = self.data[index]
        question = 'entity1: {} entity2: {}'.format(*example['target_pair']).lower()
        target = example['target']
        contexts = []
        for ctx in example['ctxs'][:self.n_context]:
            path = [ctx[0]['e1']]
            sents = []
            for i, tri in enumerate(ctx):
                path.append(tri['e2'])
                sents.append('sentence%d: %s' % (i+1, tri['sent']))
            path = '; '.join(path)
            sents = ' '.join(sents)
            contexts.append('%s %s %s' % (question, 'path: ' + path if not self.no_path else '', sents if not self.no_sent else ''))
        
        if len(contexts) < self.n_context:
            if self.duplicate_sample:
                while len(contexts) < self.n_context:
                    append_list = deepcopy(contexts[:self.n_context - len(contexts)])
                    contexts.extend(append_list)
            else:
                contexts.extend([question] * (self.n_context - len(contexts)))

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
    