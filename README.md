# Hashtags_Prediction

## Features
  * Cleaning dataset
  * Extract dataset from large stream file
  * Train CNN, RNN, RNN+CNN, neural network (models)
  * Predict hashtags with trained models

## Dataset description:

The dataset is compose of more than 2M tweets.
```
Tweets are json object separated by '\n'
```

### Data look like
```json
{"text": "The Evolution of the             https://t.co/frtmvBHIFdUtarSystems", "hashtags": ["IoT", "BigData", "DataScience", "Chatbot", "9and9", "Fintech", "Insurtech", "AI", "VR", "AR", "Startup"]}
{"text": "Follow the tips bellow and you could be on your way to win 20k monthly cc  @HHappyfish @busolaholloway  https://t.co/pxiGlNaJ44", "hashtags": ["OnlineHype"]}
```
>Hashtags has already been removed from text !
## How to get the dataset ?

```bash
cat data/corpus/x* > corpus.json
```

## How to clean data ?

```bash
./cleanStep.sh path_to_corpus path_to_cleaned_corpus
```

## How to train model ?

#### Dependencies
  * pyTorch (4.0.0)
  * numpy
#### Compute words occurences
```bash
python3 path_to_corpus path_to_words_occurences_dic
```
#### Train models

```bash
python3  cleaned_corpus words_occurences path_to_save_models path_to_save_vocab
```

## Predict hashtags

```bash
python3 src/predictHashtags.py model_path doc/vocab.json doc/vocabH.json text
```

## Web api
```bash
python3 src/server.py [port]
```

### Request
