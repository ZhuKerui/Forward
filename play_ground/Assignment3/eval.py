from transformers import BartForConditionalGeneration, BartTokenizer, BatchEncoding
import datasets
import torch
import tqdm

# Some global variable
dataset_dir = 'cnn_summary'

tokenizer = BartTokenizer.from_pretrained('facebook/bart')

# [Load] huggingface dataset 
ds = datasets.DatasetDict.load_from_disk(dataset_dir)
test_dataset = ds['test']

model = BartForConditionalGeneration.from_pretrained(dataset_dir)
model.eval()
device = torch.device(0)
model.to(device)

sents = []
def batch_gen(l, n):
    start = 0
    while start < len(l):
        if (start + n) <= len(l):
            yield l[start : start + n]
        else:
            yield l[start : ]
        start += n

test_list = list(batch_gen(test_dataset, 50))
with torch.no_grad():
    for data in tqdm.tqdm(test_list):
        output = model.generate(**BatchEncoding(tokenizer(data['source'], padding='max_length', max_length=700, truncation=True, return_tensors="pt")).to(device))
        sents += tokenizer.batch_decode(output)

with open('output.txt', 'w') as f_out:
    f_out.write('\n'.join([' '.join(d[7:-4].split()) for d in sents]))