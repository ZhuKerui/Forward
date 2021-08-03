import sys

sys.path.append('..')

if __name__ == '__main__':
    import pandas as pd
    from transformers import BertTokenizer, BertForSequenceClassification, BatchEncoding, AdamW
    import torch
    from typing import Iterable
    import tqdm

    # Load training and validation data
    train_df = pd.read_csv('../data/temp/1st_sent/train.csv')
    valid_df = pd.read_csv('../data/temp/1st_sent/valid.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'additional_special_tokens' : ['<HEAD_ENT>', '<TAIL_ENT>', '<DEP_PATH>']})

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(len(tokenizer))
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

    for epoch in range(4):
        loss = 0
        batch_num = 0
        for batch_df in tqdm.tqdm(batch_list):
            optim.zero_grad()
            labels = torch.tensor([1 if i == 'T' else 0 for i in batch_df.label.to_list()]).unsqueeze(1).to(device)
            inputs = BatchEncoding(tokenizer(batch_df.sent.to_list(), batch_df.pair.to_list(), padding=True, truncation=True, max_length=80, return_tensors="pt")).to(device)
            output = model(**inputs, labels=labels)
            loss += output.loss
            output.loss.backward()
            optim.step()
        print(loss / len(batch_list))

    # Save trained model
    model.save_pretrained('../data/temp/1st_sent/test1.pt')
    tokenizer.save_pretrained('../data/temp/1st_sent/test1.pt')