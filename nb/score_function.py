# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import random
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.append('..')
from tools.Models import save_model, rescale_gradients, ScoreFunctionModel_1, ScoreFunctionModel_2


# %%
# Some constants
max_path_len = 10
embed_dim = 512
dataset_file = '../data/corpus/score_func_dataset.csv'
train_file = '../data/temp/train.csv'
valid_file = '../data/temp/valid.csv'
epochs = 6
lr = 0.01
grad_norm = 10
dev_every = 5000
save_path = '../data/output/score_func/save_path_1'


# %%
# Load positive examples
df_pos = pd.read_csv(dataset_file)
df_pos['label'] = 'T'


# %%
# Create negative examples for model 1
def gen_neg_path(path:str):
    items = path.split()
    if len(items) < 2 or len(set(items)) == 1:
        return ''
    shuffled_list = random.sample(items, len(items))
    while shuffled_list == items:
        shuffled_list = random.sample(items, len(items))
    return ' '.join(shuffled_list)

df_neg = df_pos.copy()
df_neg['path'] = df_neg.apply(lambda row: gen_neg_path(row['path']), axis=1)
df_neg = df_neg[df_neg['path'] != '']
df_neg['label'] = 'F'


# %%
# Create negative examples for model 2
# df_neg = pd.concat([df_pos.path.to_frame(), 
#                     df_pos.subj.sample(frac=1).reset_index(drop=True).to_frame(), 
#                     df_pos.obj.sample(frac=1).reset_index(drop=True).to_frame()], axis=1)
# df_neg['label'] = 'F'


# %%
# Split the dataset into training and validation dataset
df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
total_num = len(df)
train_df = df[:int(total_num*0.8)]
valid_df = df[int(total_num*0.8):]
train_df.to_csv(train_file, index=False)
valid_df.to_csv(valid_file, index=False)


# %%
# Load training and validation dataset
train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)


# %%
# Generate Vocabularies
path_tokenizer = lambda x: x.split()

path_c = Counter()
for line in train_df['path']:
    path_c.update(path_tokenizer(line))
path_vocab = Vocab(path_c)

entity_c = Counter()
entity_c.update(train_df['subj'].values.tolist())
entity_c.update(train_df['obj'].values.tolist())
entity_vocab = Vocab(entity_c)


# %%
# Define and generate training and validation dataset
class MyDataset(Dataset):
    def __init__(self, corpus_file):
        self.dataset = pd.read_csv(corpus_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset['path'][idx], self.dataset['subj'][idx], self.dataset['obj'][idx], self.dataset['label'][idx]

train_dataset = MyDataset(train_file)
valid_dataset = MyDataset(valid_file)


# %%
# Define and generate dataloader
def collate_batch(batch):
   path_list, subj_list, obj_list, label_list = [], [], [], []
   for (_path, _subj, _obj, _label) in batch:
        path_list.append(torch.tensor([path_vocab.stoi[item] for item in _path.split()]))
        subj_list.append(entity_vocab.stoi[_subj])
        obj_list.append(entity_vocab.stoi[_obj])
        label_list.append(1 if _label == 'T' else -1)
   return pad_sequence(path_list, padding_value=path_vocab.stoi['<pad>']), torch.tensor(subj_list), torch.tensor(obj_list), torch.tensor(label_list)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)


# %%
# Define the training process
def train(save_path:str, path_vocab:Vocab, 
    train_iter:DataLoader, val_iter:DataLoader, 
    model_type:int=1, embed_dim:int=512, max_path_len:int=10, epochs:int=6, dev_every:int = 5000, 
    grad_norm:int = 10, lr:float=0.01, 
    entity_vocab:Vocab=None, 
    retrain:bool=False, cp_file:str=None):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 1:
        model = ScoreFunctionModel_1(device, len(path_vocab), embed_dim)
    elif model_type == 2:
        model = ScoreFunctionModel_2(device, len(path_vocab), len(entity_vocab), embed_dim)
    else:
        return
    if retrain:
        checkpoint = torch.load(cp_file)
        model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    best_train_loss = 1000
    os.makedirs(save_path)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.SGD(params, lr=lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=10, verbose=True, threshold=0.001)
    iterations = 0

    for epoch in range(epochs):
        train_accu_loss = 0
        train_cnt = 0
        
        for path_, subj_, obj_, label_ in tqdm(iter(train_iter)):
            # Switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()
            iterations += 1
                
            # forward pass
            if model_type == 1:
                answer, loss = model(path_.to(device), path_vocab.stoi['<pad>'], label_.to(device))
            else:
                answer, loss = model(path_.to(device), path_vocab.stoi['<pad>'], subj_.to(device), obj_.to(device), label_.to(device))
            
            # backpropagate and update optimizer learning rate
            loss.backward()

            # grad clipping
            rescale_gradients(model, grad_norm)
            opt.step()
            
            # aggregate training error
            train_accu_loss += loss.item()
            train_cnt += 1
            
        # evaluate performance on validation set periodically
        model.eval()
        eval_accu_loss = 0
        eval_cnt = 0
        for dev_path_, dev_subj_, dev_obj_, dev_label_ in iter(val_iter):
            if model_type == 1:
                answer, loss = model(dev_path_.to(device), path_vocab.stoi['<pad>'], dev_label_.to(device))
            else:
                answer, loss = model(dev_path_.to(device), path_vocab.stoi['<pad>'], dev_subj_.to(device), dev_obj_.to(device), dev_label_.to(device))
            eval_accu_loss += loss.item()
            eval_cnt += 1
        eval_loss = eval_accu_loss / eval_cnt
        scheduler.step(eval_loss)
        
        train_loss = train_accu_loss / train_cnt

        print('train_loss: %.3f, eval_loss: %.3f' % (train_loss, eval_loss))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_model(model, save_path, train_loss, iterations, 'best_train_snapshot')
        
        save_model(model, save_path, train_loss, iterations, 'epoch_train_snapshot')

        # reset train stats
        train_accu_loss = 0
        train_cnt = 0


# %%
train(save_path=save_path, path_vocab=path_vocab, train_iter=train_dataloader, val_iter=valid_dataloader, model_type=1)


# %%



