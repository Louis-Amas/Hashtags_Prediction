import json
from sys import argv
from collections import OrderedDict
from operator import itemgetter


def computeScoreHashtags(list_tweets):
    """

    :param list_tweets: List of json object tweets
    :return: OrderedDict key = Unique hashtags and value = occurences
    """
    dic = dict()
    for tweet in list_tweets:
        for hashtag in tweet['hashtags']:
            if hashtag in dic.keys():
                dic[hashtag] += 1
            else:
                dic[hashtag] = 1
    return OrderedDict(sorted(dic.items(), key=itemgetter(1), reverse=True))


def saveOnlySelectedTweet(scores, list_tweets, save_path, min_hashtags=200):
    """
        Delete all tweets where their hashtags is less frequent that min_hashtags
    :param scores: OrderedDict
    :param list_tweets: List of json object tweets
    :param save_path: path output file
    :param min_hashtags: min freq for hashtags
    """
    with open(save_path, 'w') as fOut:
        for i, tweet in enumerate(list_tweets):
            score = scores[tweet['hashtags'][0]]
            if  score < min_hashtags or score > max_hashtags:
                continue
            fOut.write(json.dumps(tweet) + '\n')


if __name__ == '__main__':
    """
        Delete line with non frequent hashtags
    """
    if len(argv) < 4:
        print(argv[0], ' file.json minFreq newFile.json')
        exit(1)

    fic_in = argv[1]
    min_freq = int(argv[2])
    fic_out = argv[3]
    with open(fic_in, 'r') as f:
        list_tweets = []
        for line in f.readlines():
            tweet = json.loads(line)
            list_tweets.append(tweet)
        scores = computeScoreHashtags(list_tweets)
        saveOnlySelectedTweet(scores, list_tweets, fic_out, min_freq)
