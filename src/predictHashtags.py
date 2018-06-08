import torch
import json
from sys import argv

from trainNN import RNN3
from trainNN import transform_text



def getHashtags(predict, rev_vocabh):
    pre = predict.detach().cpu().numpy()
    bestHashtags = pre.argsort()[-3:][::-1]
    preds = []
    for i in range(0, 3):
      preds.append(rev_vocabh[bestHashtags[i]])

    return preds

def printPreds(text, preds):
    print('X:', text + '\n' + '-' * 30)
    for pred in preds:
        print(pred)
    print('')


def predict(modelPath, vocabPath, vocabHPath, tweets_text, authorized_words):
    cuda = torch.cuda.is_available()
    seqLen = 15

    with open(vocabPath, 'r') as f:
        vocab = json.loads(f.read())
    with open(vocabHPath, 'r') as f:
        vocabH = json.loads(f.read())

    rev_vocab = {vocab[key]: key  for key in vocab }
    rev_vocabh = {vocabH[key]: key  for key in vocabH }

    if cuda:
        model = torch.load(modelPath, map_location={'cuda:0': 'cpu'})
    else:
        model = torch.load(modelPath)

    texts_vec = torch.LongTensor(transform_text(tweets_text, seqLen, vocab, authorized_words))
    bo = model.forward(texts_vec)
    texts_preds = []
    for i in range(bo.shape[0]):
        texts_preds.append(getHashtags(bo[i], rev_vocabh))
    return texts_preds
        


if __name__ == '__main__':
    if len(argv) < 6:
        print(argv[0], 'path_to_model path_vocab path_vocabh tweet_text wordoc')
        exit(1)

    modelPath = argv[1]
    vocabPath = argv[2]
    vocabHPath = argv[3]
    with open(argv[4]) as f:
        authorized_words = json.loads(f.read())


    tweets_text = argv[5:]
    texts_preds = predict(modelPath, vocabPath, vocabHPath, tweets_text, authorized_words)
    for i, preds in enumerate(texts_preds):
        printPreds(tweets_text[i], preds)
