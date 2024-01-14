import torch
import torch.nn as nn

class mymodel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim = 128, hidden_dim_1 = 64, hidden_dim_2 = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm_1 = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim_1, num_layers = 2, dropout = 0.3)
        self.lstm_2 = nn.LSTM(input_size = hidden_dim_1, hidden_size = hidden_dim_2, num_layers = 1, dropout = 0.2)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x):
        x = self.embedding(x)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        #x = torch.squeeze(x,dim=0)
        x = self.softmax(x)
        return x
