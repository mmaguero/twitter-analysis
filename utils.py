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
import l3 # ParaMorfo for gn
import pandas as pd

# load custom lemma
lemmatizer = SpacyCustomLemmatizer()
nlp.add_pipe(lemmatizer, name="lemmatizer", after="tagger")
    

def preprocess(tweet, ascii=True, ignore_rt_char=True, ignore_url=True,
               ignore_mention=True, ignore_hashtag=True,
               letter_only=True, remove_stopwords=True, min_tweet_len=3,
               content_words=True, lang='es'):
               
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
  #
  gn_early_exit = ["nicaragua"] # lang_detect interprets gn

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
    if lang != 'es' and token in gn_early_exit:
      return ''
    if remove_stopwords and lang == 'es' and token in sword:
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
    #POS 
    if content_words and lang == 'es' and pos not in ["NOUN","PROPN","ADV","ADJ","VERB"]: # es
      continue
    if content_words and lang != 'es' and get_tag(token) not in ['n','v','adj','adv'] and pos not in ["NOUN","PROPN","ADV","ADJ","VERB"]: # gn
      continue
    #
      
    token = t.lemma_ if lang == 'es' else get_stem(token, True)
    res += token,

  #min_tweet_len
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


def evaluate_model(file_name, date, n_iter, scope, lang, n_eval=5):
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
  name = "evaluate_model_{}_{}iter_{}eval_{}_{}.png".format(date, n_iter, n_eval, scope, lang)
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
    
    
def get_latam_countries_homonyms(file_name, homonym=False):
    # http://lanic.utexas.edu/subject/countries/indexesp.html
    # https://www.cookingideas.es/ciudades-homonimas-20131114.html
    #
    list_ = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line.strip())<1 or not line:
                continue
            if "(" in line:
                if homonym:
                    line = line.strip().split("(")[0] # get the first elemente of: city (country) vs. city (spain)
                else:
                    continue
            list_.append(line.strip().lower())
    f.close()
    #
    return list_


def get_spain_places(file_name):
    # https://raw.githubusercontent.com/social-link-analytics-group-bsc/tw_coronavirus/master/data/places_spain.csv
    df = pd.read_csv(file_name)
    list_ = df['comunidad autonoma'].tolist()
    list_ += df['provincia'].tolist()
    list_ += df['ciudad'].tolist()
    list_ = set([str(x).strip().lower() for x in set(list_) if len(str(x).strip())>0 and x is not None or str(x).strip().lower() not in ['nan','na','none']])
    return list(list_)
    
    
def get_stem(text, mixed=False):
    stem = []
    doc = nlp(text)
    for token in doc:
        token_ = token.lemma_
        if len(l3.anal('gn', token.text, raw=True)) > 0:
            try:
                token_ = l3.anal('gn', token.text, raw=True)[0][0] # use paramorfo word root
            except:
                token_ = token.text
        elif mixed:
            token_ = token.lemma_ # use spacy lemmatizer
        token = token_ if token.is_punct else " "+token_ # for delete extra space: punctuaction
        stem.append(token)
    return "".join(stem).replace("<","").strip()
    

def get_tag(word):
    tag = 'u'
    if len(l3.anal('gn', word, raw=True)) > 0:
        try:
            tag = l3.anal('gn', word, raw=True)[0][1]['pos'] # use paramorfo for pos 
        except:
            tag = 'x' # undefined or missing 
    return tag
