# Twitter analysis


## virtualenv

First create a virtual environment in the root dir by running:

`python3 -m venv venv`

then activate the virtual env with

`source venv/bin/activate`

(to get out of the virtualenv, run `deactivate`)


## Dependencies

install all the dependencies with

`pip install -r requirements.txt`

also make sure to download nltk's corpus by running those line in python
interpreter:

```python
import nltk
nltk.download()
```
and spacy model:

```bash
python -m spacy download es_core_news_sm
```
and spacy custom lemmatizer files:

```bash
python -m spacy_spanish_lemmatizer download wiki
```

(for language detection go to this [repo](https://github.com/mmaguero/lang_detection))

## Credentials

Rename `sample_credentials.json` to `credentials.json`, and fill in the four
credentials from your twitter app.


## Real-time twitter trend discovery 

<span style="color:gold">(Not tested in this fork)</span> Run 

`bokeh serve --show real-time-twitter-trend-discovery.py --args <tw>
<top_n_words> <*save_history>`, 

where `<tw>` and `<top_n_words>` are arguments
representing within what time window we treat tweets as a batch, and how many
words with highest idf scores to show, while `<*save_history>` is an optional
boolean value indicating whether we want to dump the history. Make sure API
credentials are properly stored in the credentials.json file.


## Topic modeling and t-SNE visualization: 20 Newsgroups

<span style="color:gold">(Not tested in this fork)</span> To train a topic model and visualize the news in 2-D space, run

`python topic_20news.py --n_topics <n_topics> --n_iter <n_iter>
--top_n <top_n> --threshold <threshold>`,
  
 where `<n_topics>` being the number
of topics we select (default 20), `<n_iter>` being the number of iterations
for training an LDA model (default 500), `<top_n>` being the number of top
keywords we display (default 5), and `<threshold>` being the threshold
probability for topic assignment (default 0.0).


## Scrape tweets and save them to disk

<span style="color:gold">(Not tested in this fork)</span> To scrape tweets and save them to disk for later use, run

`python scrape_tweets.py`. 

If the script is interrupted, just re-run the same
command so new tweets collected. The script gets ~1,000 English tweets per min,
or 1.5 million/day.

Make sure API credentials are properly stored in the credentials.json file.


## Topic modeling and t-SNE visualization: tweets

First make sure you accumulated some tweets (in this fork, we prefer https://github.com/Jefferson-Henrique/GetOldTweets-python and save it in CSV format), then run 

`python topic_tweets.py
--raw_tweet_dir <raw_tweet_dir> --num_train_tweet <num_train_tweet>
--n_topics <n_topics> --n_iter <n_iter> --top_n <top_n> --threshold <threshold>
--num_example <num_example> --start_date <start_date> --end_date <end_date> 
--scope <scope> --lang <lang> --eval_n_topics <eval_n_topics>`

where `<raw_tweet_dir>` being a folder containing
raw tweet files, `<num_train_tweet>` being the number of tweets we use for
training an LDA model, `<n_topics>` being the number of topics we select
(default 20), `<n_iter>` being the number of iterations for training an LDA
model (default 1500), `<top_n>` being the number of top keywords we display
(default 8), `<threshold>` being the threshold probability for topic assignment
(default 0.0), and `<num_example>` being number of tweets to show on the plot
(default 5000). The same for `topic_profiles.py`.

Extra params for `topic_tweets.py`: `<start_date>`, `<end_date>` for filter the data, and `<scope>`, for merge with a CSV file with Spain users (default SPA).
Also `<lang>`, for filter by language (es [stable], es_gn and gn [pre-alfa]) and `<eval_n_topics>`, if you want to evaluate the optimal numbers of topics...

### Data input
4 .csv files:
1. tweets file, with columns: 'tweet_id','tweet','date','user_id'
1. lang detected file, with columns: 'tweet_id','lang'
1. user file of particular location (Spain for us), with column: 'id_str' (then merge with 'user_id') 
1. and a extra file to check locations.

## References
1. https://github.com/lda-project/lda
1. https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
1. https://datascience.blog.wzb.eu/2017/11/09/topic-modeling-evaluation-in-python-with-tmtoolkit/
1. https://github.com/WZBSocialScienceCenter/tmtoolkit
1. https://github.com/starry9t/TopicLabel
1. https://towardsdatascience.com/%EF%B8%8F-topic-modelling-going-beyond-token-outputs-5b48df212e06
