import json
from sys import argv
import re

def process(status):
    if 'delete' in status.keys():
        return
    hashtags = status['entities']['hashtags']
    replaced = re.sub('#[^\ \t\n]+', '', status['text'])
    replaced = replaced.replace('\n', ' ')
    if len(hashtags) == 0 or status['lang'] != 'en':
        return None
    hashtags = [hashtag[key] for hashtag in hashtags for key in hashtag if key == 'text']
    dic = dict()
    dic['text'] = replaced
    dic['hashtags'] = hashtags
    return dic


if __name__ == '__main__':
    if len(argv) < 2:
        print(argv[0], ' files.json n.json')
        exit(1)

    json_files = argv[1:-1]
    out_file = argv[len(argv) - 1]

    for fic in json_files:
        with open(fic, 'r') as file:
            for line in file.readlines():
                list_tweet = []
                jsonObj = json.loads(line)
                tw = process(jsonObj)
                if tw == None:
                    continue
                list_tweet.append(tw)
                with open(out_file , 'a+') as fOut:
                    for tweet in list_tweet:
                        fOut.write(json.dumps(tweet) + '\n')



