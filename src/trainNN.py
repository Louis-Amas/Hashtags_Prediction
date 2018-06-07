import numpy as np
import sys
from sys import argv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import json


def read_data(file_path):
    texts = []
    hashtags_by_text = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            json_obj = json.loads(line)
            texts.append(json_obj['text'])
            hashtags_by_text.append(json_obj['hashtags'])

    return hashtags_by_text, texts


def transform_text(features, input_seq_size, vocab_to_int, authorized_words):
    X = np.zeros((len(features), input_seq_size))
    for j, text in enumerate(features):
        words = text.split()
        text_int = np.zeros(input_seq_size)
        for i in range(input_seq_size):
            if i <= len(words) - 1:
                if words[i] in authorized_words:
                    text_int[i] = vocab_to_int[words[i]]
                else:
                    text_int[i] = vocab_to_int['<UNK>']
            else:
                text_int[i] = vocab_to_int['<eos>']
        X[j] = text_int
    return X

def getFeatures(features, authorized_words):
   
    vocab = set()
    vocab.add('<eos>')
    vocab.add('<UNK>')
    for text in features:
        words = text.split()
        for word in words:
            if word in authorized_words:
                vocab.add(word)

    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    rev_vocab = {vocab_to_int[key]: key for key in vocab_to_int}
    X = transform_text(features, input_seq_size, vocab_to_int, authorized_words)
    return vocab_to_int, rev_vocab, X


def getLabels(labels):
    hashtags_set = set()
    for hashtags_in_text in labels:
        for hashtag in hashtags_in_text:
            hashtags_set.add(hashtag)

    hashtags_vocab_to_int = {hashtag: i for i, hashtag in enumerate(hashtags_set)}
    rev_hashtags_vocab_to_int = {hashtags_vocab_to_int[key]: key for key in hashtags_vocab_to_int}
    Y = np.zeros((len(labels), hashtags_seq_size))

    for j, hashtags in enumerate(labels):
        Y[j] = hashtags_vocab_to_int[hashtags[0]]
    return Y, hashtags_vocab_to_int, rev_hashtags_vocab_to_int

def perf(model, loader, cuda):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = correct = num = 0
    for x, y in loader:
        with torch.no_grad():
            if cuda:
                y_scores = model(Variable(x).cuda())
                loss = criterion(y_scores, Variable(y).cuda())
            else:
                y_scores = model(Variable(x))
                loss = criterion(y_scores, Variable(y))
        y_pred = torch.max(y_scores, 1)[1]
        correct += torch.sum(y_pred.data == y).item()
        total_loss += loss.data.item()
        num += len(y)
    return total_loss / num, correct


def train(model, epochs, cuda, model_save_path, quiet=False):
    # Selection de la fonction de coût
    criterion = nn.CrossEntropyLoss()
    
    
    for epoch in range(epochs):
        
        if epoch > 15:
            learning_rate = 0.00001
        elif epoch >= 10:
            learning_rate = 0.0001
        else:
            learning_rate = 0.001
        
        optim = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        line = 'Epoch: ' + str(epoch) + ' lr: ' + str(learning_rate) + ' '
        sys.stdout.write(line)

        model.train()
        total_loss = correct = num = 0
        cpt = 0
        for X, y in train_loader:
            if cuda:
                y_scores = model(Variable(X).cuda())
                loss = criterion(y_scores, Variable(y).cuda())
            else:
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
            cpt += 1


        print('\nCorrect train: ', round(correct / len(X_train), 5))
        loss_valid, correct_count_test = perf(model, valid_loader, cuda)
        torch.save(model, model_save_path + '/mod' + str(epoch) + '.mod')
        print('Correct test:', round(correct_count_test / len(X_valid),5), '\n')
        print( '-' * 50)


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




if __name__ == '__main__':
    if len(argv) < 5:
        print(argv[0], 'cleaned_corpus words_occurences path_to_save_models path_to_save_vocab')
        exit(1)

    cuda = torch.cuda.is_available()


    input_seq_size = 10
    embed_size = 128
    hashtags_seq_size = 1
    hidden_size = 130
    batch_size = 100
    print('Loading data...')
    hashtags_by_text, texts = read_data(argv[1])

    model_save_path = argv[3]
    doc_save_path = argv[4]

    with open(argv[2]) as f:
        authorized_words = json.loads(f.read())

    print('Format features...')

    vocab_to_int, rev_vocab, X = getFeatures(texts, authorized_words.keys())

    with open(doc_save_path + '/vocab.json', 'w') as f:
        f.write(json.dumps(vocab_to_int))

    print('Format targets...')
    Y, hashtags_vocab_to_int, rev_hashtags_vocab_to_int = getLabels(hashtags_by_text)


    with open(doc_save_path + '/vocabH.json', 'w') as f:
        f.write(json.dumps(hashtags_vocab_to_int))

    nb_examples = X.shape[0] - 1

    test_size = int(round(nb_examples / 5, 0))
    train_size = int(nb_examples - test_size)

    print ('Create training and test samples')
    Y = Y.reshape((Y.shape[0]))
    if cuda:
        X = torch.LongTensor(X).cuda()
        Y = torch.LongTensor(Y).cuda()
    else:
        X = torch.LongTensor(X)
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
        model = RNN3().cuda()
    else:
        model = RNN3()

    print('Start training...\n')
    train(model, 25, cuda, model_save_path)
