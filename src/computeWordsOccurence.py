import json
from sys import argv
from collections import OrderedDict
from operator import itemgetter
import heapq
from operator import itemgetter


def computeScoreWords(list_tweets, max_words):
    """

    :param list_tweets in lower case:
    :return dic of words occurences

    """
    dic = dict()
    print('Compute words occurences')
    for i, tweet in enumerate(list_tweets):

        for word in tweet['text'].split():
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    od = OrderedDict(sorted(dic.items(), key=itemgetter(1), reverse=True))
    topitems = heapq.nlargest(max_words, od.items(), key=itemgetter(1))
    return dict(topitems[:max_words])


if __name__ == '__main__':
    """
        Compute n most common hashtags
    """
    if len(argv) < 3:
        print(argv[0], ' hmWords file.json newFile.json')
        exit(1)

    max_words = int(argv[1])
    fic_in = argv[2]
    fic_out = argv[3]

    with open(fic_in, 'r') as f:
        print('Loading data in ram...')
        list_tweets = []
        for line in f.readlines():
            list_tweets.append(json.loads(line))

    dic = computeScoreWords(list_tweets, max_words)

    with open(fic_out, 'w') as f:
        f.write(json.dumps(dic))
