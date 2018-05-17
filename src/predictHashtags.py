import torch
import json
from sys import argv
import numpy as np
from trainNN import RNN


def getHashtags(predict):
    pre = predict.detach().cpu().numpy()
    bestHashtags = pre.argsort()[-3:][::-1]
    for i in range(len(bestHashtags[0]) - 3, len(bestHashtags[0])):
      print('Preds ', ':', rev_vocabh[bestHashtags[0][i]])
    #return rev_vocabh[torch.argmax(predict,1).item()]


def transformText(text, seqLen, vocab, cuda):
    textInInt = np.zeros((1, seqLen))
    words = text.split()
    for i in range(0, seqLen):
        if i <= len(words) - 1:
            if words[i] not in vocab.keys():
                textInInt[0][i] = vocab['<UNK>']
            else:
                textInInt[0][i] = vocab[words[i]]
        else:
            textInInt[0][i] = vocab['<eos>']
    return torch.LongTensor(textInInt)

if __name__ == '__main__':
    if len(argv) < 6:
        print(argv[0], 'path_to_model path_vocab path_vocabh tweet_text cuda')
        exit(1)

    modelPath = argv[1]
    vocabPath = argv[2]
    vocabHPath = argv[3]
    tweet_text = argv[4].lower()
    if argv[5] == 'cuda':
        cuda = True
    else:
        cuda = False

    seqLen = 15

    with open(vocabPath, 'r') as f:
        vocab = json.loads(f.read())
    with open(vocabHPath, 'r') as f:
        vocabH = json.loads(f.read())

    rev_vocab = {vocab[key]: key  for key in vocab }
    rev_vocabh = {vocabH[key]: key  for key in vocabH }

    if cuda:
        torch.load(modelPath, map_location={'cuda:0': 'cpu'})
    else:
        model = torch.load(modelPath)


    print('X:', tweet_text)
    text_vec = transformText(tweet_text, seqLen, vocab, cuda)
    text_vec.reshape([1, seqLen])
    bo = model.forward(text_vec)
    getHashtags(bo)
