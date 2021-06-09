import torch
from torch import nn
from torch.nn.functional import logsigmoid
from torch import sigmoid
import os
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import glob

class MLPPairEncoder(nn.Module):
    def __init__(self, embed_dim:int=512, dropout:float=0.1):
        super(MLPPairEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.nonlinearity  = nn.ReLU()
        self.mlp = nn.Sequential(self.dropout, nn.Linear(3 * embed_dim, embed_dim), self.nonlinearity, self.dropout, nn.Linear(embed_dim, embed_dim))
    
    def forward(self, subjects:torch.Tensor, objects:torch.Tensor):
        '''
        Args
            subjects (torch.Tensor): (N, E)
            objects (torch.Tensor): (N, E)

        Returns
            torch.Tensor: (N, E)

        N is the size of the batch and E is the length of the embedding
        '''
        return self.mlp(torch.cat([subjects, objects, subjects * objects], dim=-1))


def gen_src_mask(length:int):
    src_mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    return src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))

class MyTransformerEncoder(nn.Module):
    def __init__(self, device:torch.device, embed_dim:int=512, nhead:int=8, nlayer:int=6):
        super(MyTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.device = device

    def forward(self, src:torch.Tensor, pad_mask:torch.Tensor):
        '''
        Args:
            src (torch.Tensor): (S, N, E)
            pad_mask (torch.Tensor): (N, S)

        Returns:
            torch.Tensor: (N, E)

        S is the length of the sequence, N is the size of the batch and E is the length of the embedding
        '''
        return torch.mean(self.transformer_encoder(src, gen_src_mask(src.size()[0]).to(self.device), pad_mask), dim=0)


class ScoreFunctionModel_1(nn.Module):
    def __init__(self, device:torch.device, path_vocab_size:int, embed_dim:int):
        super(ScoreFunctionModel_1, self).__init__()
        self.path_embedding = nn.Embedding(path_vocab_size, embed_dim, sparse=True)
        self.path_encoder = MyTransformerEncoder(device=device, embed_dim=embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.__init_weights()

    def __init_weights(self):
        initrange = 0.5
        self.path_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, path_idx, pad_idx, label):
        path_embs = self.path_embedding(path_idx)
        pad_mask = path_idx == pad_idx
        path_embed = self.path_encoder(path_embs, pad_mask.T)

        scores = self.fc(path_embed).sum(-1)
        return sigmoid(scores), -logsigmoid(scores * label).sum()


class ScoreFunctionModel_2(nn.Module):
    def __init__(self, device:torch.device, path_vocab_size:int, ent_vocab_size:int, embed_dim:int):
        super(ScoreFunctionModel_1, self).__init__()
        self.path_embedding = nn.Embedding(path_vocab_size, embed_dim, sparse=True)
        self.ent_embedding = nn.Embedding(ent_vocab_size, embed_dim, sparse=True)
        
        self.path_encoder = MyTransformerEncoder(device=device, embed_dim=embed_dim)
        self.ent_encoder = MLPPairEncoder(embed_dim=embed_dim)
        
        self.__init_weights()

    def __init_weights(self):
        initrange = 0.5
        self.path_embedding.weight.data.uniform_(-initrange, initrange)
        self.ent_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, path_idx, pad_idx, subj_idx, obj_idx, label):
        path_embs = self.path_embedding(path_idx)
        pad_mask = path_idx == pad_idx
        path_embed = self.path_encoder(path_embs, pad_mask.T)

        subj_embs, obj_embs = self.ent_embedding(subj_idx), self.ent_embedding(obj_idx)
        ent_embed = self.ent_encoder(subj_embs, obj_embs)

        scores = (path_embed * ent_embed).sum(-1)
        return sigmoid(scores), -logsigmoid(scores * label).sum()


# Some helper functions
def get_lr(optimizer:optim.Optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def rescale_gradients(model:nn.Module, grad_norm):
    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
    clip_grad_norm_(parameters_to_clip, grad_norm)


def save_model(model:nn.Module, save_path:str, loss, iterations, name):
    snapshot_prefix = os.path.join(save_path, name)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_model.pt'.format(loss, iterations)
    torch.save(model.state_dict(), snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

