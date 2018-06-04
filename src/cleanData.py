from sys import argv
import json
import re

"""
    Clean data:
        :param: ficIn ficOut
        Deletes all hashtags, urls links, hashtags in text and RT
"""

do_add = lambda s, x:  len(s) != (s.add(x) or len(s))

if __name__ == '__main__':
    if len(argv) < 3:
        print(argv[0], ' file.json out.json')
        exit(1)

    texts_set = set()
    fic = open(argv[1], 'r')
    fi = open(argv[2], 'w')
    for line in fic.readlines():
        json_obj = json.loads(line)
        if len(json_obj['text'].split()) < 4:
            continue
        if not do_add(texts_set, json_obj['text']):
            continue
        json_obj['text'] = re.sub('@[^\ \t\n"]+', '', json_obj['text'])
        json_obj['text'] = re.sub('https[^\ \t\n"]+', '', json_obj['text'])
        json_obj['text'] = re.sub('#[^\ \t\n"]+', '', json_obj['text'])
        json_obj['text'] = json_obj['text'].replace('rt ', '')
        fi.write(json.dumps(json_obj) + '\n')
