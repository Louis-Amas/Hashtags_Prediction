import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab_to_int), embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.decision = nn.Linear(hidden_size * 2 * 1, len(hashtags_vocab_to_int))

    def forward(self, x):
        embed = self.embed(x)
        output, hidden = self.rnn(embed)
        cur = self.decision(hidden.transpose(0, 1).contiguous().view(x.size(0), -1))
        return cur

class RNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab_to_int), embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True)
        self.decision = nn.Linear(hidden_size * 2 * 1, len(hashtags_vocab_to_int))

    def forward(self, x):
        embed = self.embed(x)
        output, hidden = self.rnn(embed)
        cur = self.decision(hidden.transpose(0, 1).contiguous().view(x.size(0), -1))
        return cur


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab_to_int), embed_size)
        self.conv = nn.Conv1d(embed_size, hidden_size, kernel_size=2)
        self.decision = nn.Linear(hidden_size, len(hashtags_vocab_to_int))

    def forward(self, x):
        embed = self.embed(x)
        conv = F.relu(self.conv(embed.transpose(1, 2)))
        pool = F.max_pool1d(conv, conv.size(2))
        return self.decision(pool.view(x.size(0), -1))


class CNN_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab_to_int), embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(embed_size, hidden_size, kernel_size=3 )
        self.decision = nn.Linear(hidden_size, len(hashtags_vocab_to_int))


    def forward(self, x):
        embed = self.embed(x)
        output, hidden = self.rnn(embed)
        conv = F.relu(self.conv(output.transpose(1, 2)))
        pool = F.max_pool1d(conv, conv.size(2)) 
