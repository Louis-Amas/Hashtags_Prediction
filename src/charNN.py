import numpy as np
from sys import argv, stdout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import preprocess as prep
import json
import sys


def read_data(file_path):
    texts = []
    hashtags_by_text = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            json_obj = json.loads(line)
            texts.append(json_obj['text'])
            hashtags_by_text.append(json_obj['hashtags'])

    return hashtags_by_text, texts

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embed = nn.Embedding(len(vocab_dic), embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True)
        self.decision = nn.Linear(hidden_size * 2 * 1, len(vocab_hashtags))

    def forward(self, x):
        # print(x.shape)
        # embed = self.embed(x)
        # print(embed.shape)
        output, hidden = self.rnn(x)
        cur = self.decision(hidden.transpose(0, 1).contiguous().view(x.size(0), -1))
        return cur


def perf(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = correct = num = 0
    for x, y in loader:
        with torch.no_grad():
            y_scores = model(Variable(x))
            loss = criterion(y_scores, Variable(y))
        y_pred = torch.max(y_scores, 1)[1]
        correct += torch.sum(y_pred.data == y).item()
        total_loss += loss.data.item()
        num += len(y)
    return total_loss / num, correct


def train(model, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        if epoch > 15:
            learning_rate = 0.0001
        elif epoch >= 8:
            learning_rate = 0.001
        else:
            learning_rate = 0.01
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        total_loss = correct = num = 0
        line = 'Epoch: ' + str(epoch) + ' lr: ' + str(learning_rate) + ' '
        cpt = 0
        for X, y in train_loader:
            y_scores = model(Variable(X))
            loss = criterion(y_scores, Variable(y))
            loss.backward()

            optim.step()
            y_pred = torch.max(y_scores, 1)[1]
            correct += torch.sum(y_pred.data == y).item()
            total_loss += loss.data.item()
            num += len(y)

            if cpt % 10 == 0:            
                sys.stdout.write('\r')
                sys.stdout.write(line + '[' + str(num) + ',' + str(len(X_train)) + ']')
                sys.stdout.flush()
            cpt +=1

        sys.stdout.write('\n')
        print('Epoch: ', epoch)
        print('Total loss train:', total_loss / len(X_train), '\nCorrect train: ', correct / len(X_train))
        loss_valid, correct_count_test = perf(model, valid_loader)
        print('Total loss valid:', loss_valid)
        print('Correct test:', correct_count_test / len(X_valid))



if __name__ == '__main__':
    if len(argv) < 4:
        print(argv[0], 'cleaned_corpus path_to_save_models path_to_save_vocab')
        exit(1)

    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False


    dataset_size = 5000
    test_size = 500
    train_size = dataset_size - test_size

    input_seq_size = 150
    batch_size = 100
    print('Loading data...')
    hashtags_by_text, texts = read_data(argv[1])

    model_save_path = argv[2]
    doc_save_path = argv[3]

    print('Processing data...')

    vocab_dic, rev_vocab = prep.create_char_vocab(texts)

    vocab_hashtags, rev_vocab_hashtags = prep.create_word_vocab(hashtags_by_text)

    embed_size = len(vocab_dic)
    hidden_size = len(vocab_dic)

    X = prep.process_text_to_char(texts, vocab_dic, input_seq_size)
    Y = prep.process_label(hashtags_by_text, vocab_hashtags)


    X = X.astype(np.uint8)

    if cuda:
        X = torch.LongTensor(X).cuda()
        Y = torch.LongTensor(Y).cuda()
    else:
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_valid = X[train_size:]
    Y_valid = Y[train_size:]
    train_set = TensorDataset(X_train, Y_train)
    valid_set = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    if cuda:
        model = RNN().cuda()
    else:
        model = RNN()

    print('Start training...')
    train(model, 10)
