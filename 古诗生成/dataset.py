import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset

class Tokenizer:
    def __init__(self, token2id_dict):
        self.token2id_dict = token2id_dict
        self.id2token_dict = {}
        for key, value in self.token2id_dict:
            self.id2toke_dict[value] = key
        self.vocab_size = len(self.token2id_dict)
    
    def id2token(self, id):
        return self.id2token_dict[id]
    
    def token2id(self, token):
        if token in self.token2id_dict.keys():
            return self.token2id_dict[token]
        else:
            return self.token2id_dict['[UNK]']
    
    def encode(self, s):
        id_list = [self.token2id_dict['[CLS]']]
        for token in s:
            id_list.append(self.token2id(token))
        id_list.append(self.token2id_dict['[SEP]'])
        return id_list
    
    def decode(self,id_list):
        special_token = ['[CLS]','[SEP]']
        s = []
        for id in id_list:
            token = self.id2token(id)
            if token in special_token:
                continue
            s.append(token)
        return s.join('')
    
##---------------------------------------------------
# function to generate text idlist
    
def padding(text,L=64):
    if len(text) > L:
        res = list(text[:L])
    else:
        res = list(text) + ['[PAD]']*(L-len(text))
    return res
        
def get_id_list(L = 64, filepath = 'poetry.txt'):
    f = open(filepath) 

    data_list = []
    token_count_dict = {}           
    line = f.readline() 

    count = 0
    while line:  
        line = f.readline()
        if '：' in line:
            line = line.replace('：',':')
        if line.count(':') != 1:
            continue
        if line[-1] == '\n':
            line = line[:-1]
            
        main_part = line.split(':')[1]
        data_list.append(main_part)
        token_list = list(main_part)
        for token in token_list:
            if token in token_count_dict.keys():
                token_count_dict[token] += 1
            else:
                token_count_dict[token]  = 1
    f.close() 

    _tokens = [(token, count) for token, count in token_count_dict.items() if count >= 4]
    # 按词频排序
    _tokens = sorted(_tokens, key=lambda x: -x[1])

    token2id_dict = {}
    token2id_dict['[UNK]'] = 0
    token2id_dict['[CLS]'] = 1
    token2id_dict['[SEP]'] = 2
    token2id_dict['[PAD]'] = 3
    for i,_ in enumerate(_tokens):
        token2id_dict[_[0]] = i+4
    
    tokenizer = Tokenizer(token2id_dict)

    id_list = []
    for text in data_list:
        data = padding(text,L)
        id = tokenizer.encode(data)
        id_list.append(id)
    
    id_list = np.array(id_list)
    id_list = torch.from_numpy(id_list)
    return id_list, tokenizer



# -----------------------------------------------
#  generate a dataloader
def text_data_loader(id_list, batch_size, shuffle=True):
    data_loader = DataLoader(dataset= id_list,batch_size= batch_size,shuffle=shuffle)
    return data_loader
'''
class dataloader(Dataset):
    def __init__(self, flag='train') -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'

        if self.flag == 'train':
            self.data = args.data_train
        else:
            self.data = args.data_val

    def __getitem__(self, index: int):
        val = self.data[index]

        if val > 8:
            label = 1
        else:
            label = 0

        return torch.tensor(label, dtype=torch.long), torch.tensor([val], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)
'''
