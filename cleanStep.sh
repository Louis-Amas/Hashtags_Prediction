#!/usr/bin/env bash

corpus=$1;
out=$2;
min_freq_hash=$3;

echo 'To lower...';
python3 src/toLower.py $corpus 1.tmp;
echo 'Deleting hashtags, usernames, url links, RT and duplicate from text...';
python3 src/cleanData.py 1.tmp 2.tmp;
echo 'Deleting tweets with uncommon hashtags...';
python3 src/deleteUnCommonHashtags.py 2.tmp $min_freq_hash $out;
rm *.tmp;
