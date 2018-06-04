import numpy as np
from sys import argv, stdout
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


def getFeatures(features, authorized_words):
    X = np.zeros((len(features), input_seq_size))
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
        # hashtags_seq = np.zeros(hashtags_seq_size)
        #
        # for i in range(hashtags_seq_size):
        #     if i <= len(hashtags) - 1:
        #         hashtags_seq[i] = hashtags_vocab_to_int[hashtags[i]]
        #     else:
        #         hashtags_seq[i] = -1
        # Y[j] = hashtags_seq
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
    # Selection de l'optimisateur ici SGD (Gradient Descent)
    optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    # Pour le nombre d'epoch spécifié faire

    for epoch in range(epochs):
        # Met le modèle en mode entrainement
        model.train()
        total_loss = correct = num = 0
        line = ''
        cpt = 0
        for X, y in train_loader:
            if cuda:
                y_scores = model(Variable(X).cuda())
                loss = criterion(y_scores, Variable(y).cuda())
            else:
                # Fait une prédiction (forward)
                y_scores = model(Variable(X))
                # Compare la prédiction a la valeur attendu calcul l'erreur (loss)
                loss = criterion(y_scores, Variable(y))
                # Calcul le gradient

            loss.backward()
            # Change les paramètres avec le gradient calculé
            optim.step()
            y_pred = torch.max(y_scores, 1)[1]
            correct += torch.sum(y_pred.data == y).item()
            total_loss += loss.data.item()
            num += len(y)

            if not quiet and cpt % 100 == 0:
                print('Epoch: ', epoch, ' ', num, ' / ', len(X_train), line)
            cpt += 1


        print('Epoch: ', epoch)
        print('Total loss train:', total_loss / num, '\nCorrect train: ', correct / len(X_train))
        loss_valid, correct_count_test = perf(model, valid_loader, cuda)
        torch.save(model, model_save_path + '/mod' + str(epoch) + '.mod')
        print('Total loss valid:', loss_valid, '\n', 'Correct test:', correct_count_test / len(X_valid))


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab_to_int), embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False, batch_first=True)
        self.decision = nn.Linear(hidden_size * 1 * 1, len(rev_hashtags_vocab_to_int))
        #self.activation = nn.Softmax()

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
        self.decision = nn.Linear(hidden_size, len(rev_hashtags_vocab_to_int))

    def forward(self, x):
        embed = self.embed(x)
        conv = F.relu(self.conv(embed.transpose(1,2)))
        pool = F.max_pool1d(conv, conv.size(2))
        return self.decision(pool.view(x.size(0), -1))


if __name__ == '__main__':
    if len(argv) < 6:
        print(argv[0], ' cuda cleaned_corpus words_occurences path_to_save_models path_to_save_vocab')
        exit(1)

    if argv[1] == 'cuda':
        cuda = True
    else:
        cuda = False

    input_seq_size = 10
    embed_size = 128
    hashtags_seq_size = 1
    hidden_size = 130
    batch_size = 100
    print('Loading data...')
    hashtags_by_text, texts = read_data(argv[2])

    model_save_path = argv[4]
    doc_save_path = argv[5]

    with open(argv[3]) as f:
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
        model = CNN().cuda()
    else:
        model = CNN()

    print('Start training...')
    train(model, 10, cuda, model_save_path)
