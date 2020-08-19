"""
  Some utilities.
"""

from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#
#from pattern.es import singularize
from stop_words import get_stop_words
import stopwordsiso as stopwordsiso
import spacy
from spacy_spanish_lemmatizer import SpacyCustomLemmatizer
nlp = spacy.load('es_core_news_md', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
import matplotlib.pyplot as plt   # for plotting the results
plt.style.use('ggplot')
from tmtoolkit.topicmod import tm_lda
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.corpus import Corpus
from collections import defaultdict

# load custom lemma
lemmatizer = SpacyCustomLemmatizer()
nlp.add_pipe(lemmatizer, name="lemmatizer", after="tagger")
    

def preprocess(tweet, ascii=True, ignore_rt_char=True, ignore_url=True,
               ignore_mention=True, ignore_hashtag=True,
               letter_only=True, remove_stopwords=True, min_tweet_len=3,
               content_words=True):
               
  key_words = ["coronavirus","corona","virus","coronaoutbreak","covid-19","covid19","2019-ncov","2019ncov","sars-cov-2","sarscov2","cov-19","cov19","covd19","covd19"] # keywords
  sword_en = set(stopwords.words('english'))
  sword_es = set(stopwords.words('spanish'))
  stop_words_iso = set(stopwordsiso.stopwords(["es", "en"]))
  reserved_words = ["rt", "fav", "vía", "nofollow", "twitter", "true", "href", "rel"]
  stop_words_es = set(get_stop_words('es'))
  stop_words_en = set(get_stop_words('en'))
  sword = set()
  sword.update(sword_en)
  sword.update(sword_es)
  sword.update(stop_words_en)
  sword.update(stop_words_iso)
  sword.update(stop_words_es)
  sword.update(reserved_words)
  sword.update(key_words)

  if ascii:  # maybe remove lines with ANY non-ascii character
    for c in tweet:
      if not (0 < ord(c) < 127):
        return ''

  #tokens = tag(tweet.lower()) #tweet.lower().split()  # to lower, split
  doc = nlp(tweet.lower())
  res = []

  for token in doc:
    t = token
    token = t.text
    pos = t.pos_
    #POS
    # pattern: ['NN','NNP','VB','JJ','RB']:
    if content_words and pos not in ["NOUN","PROPN","ADV","ADJ","VERB"]:
      continue
    #
    if remove_stopwords and token in sword:
      continue
    if ignore_rt_char and token == 'rt':
      continue
    if ignore_url and token.startswith('https:'):
      continue
    if ignore_mention and token.startswith('@'):
      continue
    if ignore_hashtag and token.startswith('#'):
      continue
    if letter_only:
      if not token.isalpha():
        continue
    elif token.isdigit():
      token = '<num>'
    # singular only ...
    '''if pos in ['NOUN','ADJ']:
      token = singularize(token)
    elif pos in ['VERB']:
      token = t.lemma_'''
    token = t.lemma_
    res += token,

  if min_tweet_len and len(res) < min_tweet_len:
    return ''
  else:
    return ' '.join(res)


def get_tfidf(tweet_list, top_n, max_features=5000, min_df=5):
  """ return the top n feature names and idf scores of a tweets list """
  tfidf_vectorizer = TfidfVectorizer(min_df=min_df,max_features=max_features)
  tfidf_vectorizer.fit_transform(tweet_list)
  indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
  features = tfidf_vectorizer.get_feature_names()
  top_feature_name = [features[i] for i in indices[:top_n]]
  top_feautre_idf = tfidf_vectorizer.idf_[indices][:top_n]

  return top_feature_name, top_feautre_idf


def evaluate_model(file_name, date, n_iter, scope, n_eval=5):
  #
  corpus = Corpus()
  corpus.add_files(file_name, encoding='utf8')
  #
  preproc = TMPreproc(corpus)
  dtm_bg = preproc.dtm
  #
  var_params = [{'n_topics': k} for k in range(5, int(n_eval*10), n_eval)]
  #
  const_params = {
    'n_iter': n_iter,
    'random_state': 20200713  # to make results reproducible
  }
  eval_results = evaluate_topic_models(dtm_bg,
                                     varying_parameters=var_params,
                                     constant_parameters=const_params,
                                     metric=['loglikelihood', 'cao_juan_2009', 'arun_2010']#,
                                     #return_models=True
                                     )
  #
  eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
  #
  name = "evaluate_model_{}_{}iter_{}eval_{}.png".format(date, n_iter, n_eval, scope)
  plot_eval_results(eval_results_by_topics, figsize=(8, 6), metric_direction_font_size='x-small', title_fontsize='small', axes_title_fontsize='x-small')
  plt.tight_layout()
  plt.savefig('out/'+name)
  return
  
    
def buildTWDScoreDict(topic_word_matrix, id2word):
    twdsdict = defaultdict(float)
    (topicnumber,wordnumber) = np.shape(topic_word_matrix)
    for topicid in range(topicnumber):
        for wordid in range(wordnumber):
            word = id2word[wordid]
            t_prime =[topic_word_matrix[z][wordid] for z in range(topicnumber) if z != topicid]
            fenmu = max(t_prime)
            twdsdict[(topicid,word)] = topic_word_matrix[topicid][wordid]/fenmu
    print('a topic word dscore dictionary is built.')
    return twdsdict
    
    
def sentDiscriminativeScore(sentence, topicid, twdsdict):
    sentence_score = 0
    for word in sentence.split():
        sentence_score += twdsdict[(topicid,word)]
    return sentence_score/len(sentence)
