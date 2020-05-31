"""
  Train LDA model using https://pypi.python.org/pypi/lda,
  and visualize in 2-D space with t-SNE.

"""

import os
import time
import lda
import random
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
from utils import preprocess
#
import dask.dataframe as dd
import glob
import pandas as pd


if __name__ == '__main__':

  lda_base = 'lda_simple'
  if not os.path.exists(lda_base):
    os.makedirs(lda_base)

  ##############################################################################
  # cli inputs

  parser = argparse.ArgumentParser()
  parser.add_argument('--raw_tweet_dir', required=True, type=str,
                      help='a directory of raw tweet files')
  parser.add_argument('--num_train_tweet', required=True, type=int,
                      help='number of tweets used for training a LDA model')
  parser.add_argument('--n_topics', required=True, type=int, default=20,
                      help='number of topics')
  parser.add_argument('--n_iter', required=True, type=int, default=1500,
                      help='number of iteration for LDA model training')
  parser.add_argument('--top_n', required=True, type=int, default=8,
                      help='number of keywords to show for each topic')
  parser.add_argument('--threshold', required=True, type=float, default=0.0,
                      help='threshold probability for topic assignment')
  parser.add_argument('--num_example', required=True, type=int, default=5000,
                      help='number of tweets to show on the plot')
  args = parser.parse_args()

  # unpack
  raw_tweet_dir = args.raw_tweet_dir
  num_train_tweet = args.num_train_tweet
  n_topics = args.n_topics
  n_iter = args.n_iter
  n_top_words = args.top_n
  threshold = args.threshold
  num_example = args.num_example


  ##############################################################################
  # get training tweets

  num_scanned_tweet = 0
  num_qualified_tweet = 0

  all_files = glob.glob(raw_tweet_dir + "week_*_all.tsv")
  li = []
  for filename in all_files:
      print(filename)
      df = pd.read_csv(filename, index_col=None, #header=0,
                sep='\t',encoding = 'utf8',lineterminator='\n', usecols = [16,20],
                names=['user_id_str','user_description'] ,low_memory=False)
      li.append(df)

  raw_tweet_files = pd.concat(li, axis=0, ignore_index=True)
  
  raw_tweet_files.info()
  raw_tweet_files['user_id_str'] = pd.to_numeric(raw_tweet_files['user_id_str'], errors='coerce')
  raw_tweet_files["user_description"]=raw_tweet_files["user_description"].astype(str)
  raw_tweet_files = raw_tweet_files[~raw_tweet_files['user_id_str'].isnull()]
  raw_tweet_files = raw_tweet_files[~raw_tweet_files['user_description'].isnull()]
  raw_tweet_text = set(raw_tweet_files['user_description'])

  print('len', len(raw_tweet_text))

  raw_tweet = []
  processed_tweet = []
  processed_tweet_set = set()  # for quicker'item in?' check

  t0 = time.time()

  '''for f in raw_tweet_files:
    in_file = os.path.join(raw_tweet_dir, f)
    if not in_file.endswith('.txt'):  # ignore non .txt file
      continue'''
  #for t in open(in_file):
  for row in raw_tweet_text:
      num_scanned_tweet += 1
      p_t = preprocess(row)
      if p_t and p_t not in processed_tweet_set: # ignore duplicate tweets
        raw_tweet += row,
        processed_tweet += p_t,
        processed_tweet_set.add(p_t)
        num_qualified_tweet += 1

      if num_scanned_tweet % 1000000 == 0:  # progress update
        print('scanned {} tweets'.format(num_scanned_tweet))

      if num_qualified_tweet == num_train_tweet:  # enough data for training
        break

  '''if num_qualified_tweet == num_train_tweet:  # break outer loop
      break'''

  del processed_tweet_set  # free memory

  t1 = time.time()
  print('\n>>> scanned {} tweets to find {} trainable; took {} mins\n'.format(
    num_scanned_tweet, num_train_tweet, (t1-t0)/60.))

  ##############################################################################
  # train LDA

  # ignore terms that have a document frequency strictly lower than 5, 10
  cvectorizer = CountVectorizer(min_df=5)
  cvz = cvectorizer.fit_transform(processed_tweet)

  lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
  X_topics = lda_model.fit_transform(cvz)

  t2 = time.time()
  print('\n>>> LDA training done; took {} mins\n'.format((t2-t1)/60.))

  np.save('lda_simple/lda_doc_topic_{}tweets_{}topics.npy'.format(
    X_topics.shape[0], X_topics.shape[1]), X_topics)
  np.save('lda_simple/lda_topic_word_{}tweets_{}topics.npy'.format(
    X_topics.shape[0], X_topics.shape[1]), lda_model.topic_word_)
  print('\n>>> doc_topic & topic word written to disk\n')

  ##############################################################################
  # threshold and plot

  _idx = np.amax(X_topics, axis=1) > threshold  # idx of tweets that > threshold
  _topics = X_topics[_idx]
  _raw_tweet = np.array(raw_tweet)[_idx]
  _processed_tweet = np.array(processed_tweet)[_idx]

  # t-SNE: 50 -> 2D
  tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                    init='pca')
  tsne_lda = tsne_model.fit_transform(_topics[:num_example])

  t3 = time.time()
  print('\n>>> t-SNE transformation done; took {} mins\n'.format((t3-t2)/60.))

  # find the most probable topic for each tweet
  _lda_keys = []
  for i, tweet in enumerate(_raw_tweet):
    _lda_keys += _topics[i].argmax(),

  # generate random hex color
  colormap = []
  for i in range(X_topics.shape[1]):
    r = lambda: random.randint(0, 255)
    colormap += ('#%02X%02X%02X' % (r(), r(), r())),
  colormap = np.array(colormap)

  # show topics and their top words
  topic_summaries = []
  topic_word = lda_model.topic_word_  # get the topic words
  vocab = cvectorizer.get_feature_names()
  for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

  # use the coordinate of a random tweet as string topic string coordinate
  topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
  for topic_num in _lda_keys:
    if not np.isnan(topic_coord).any():
      break
    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

  # plot

  title = "t-SNE visualization of LDA model trained on {} tweets, {} topics, " \
          "thresholding at {} topic probability, {} iter ({} data points and " \
          "top {} words)".format(num_qualified_tweet, n_topics, threshold,
                                 n_iter, num_example, n_top_words)

  plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                       title=title,
                       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)

                   # create the dictionary with all the information    
  plot_dict = {
        'x': tsne_lda[:, 0],#tsne_lda[:num_example, 0],
        'y': tsne_lda[:, 1],#tsne_lda[:num_example, 1],
        'colors': colormap[_lda_keys][:num_example],
        'tweet': _raw_tweet[:num_example],#text[:num_example],
        'topic_key': _lda_keys[:num_example]
        }

  # create the dataframe from the dictionary
  plot_df = pd.DataFrame.from_dict(plot_dict)

  # declare the source    
  source = bp.ColumnDataSource(data=plot_df)

  # build scatter function from the columns of the dataframe
  plot_lda.scatter('x', 'y', color='colors', source=source)

  '''plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                   color=colormap[_lda_keys][:num_example],
                   source=bp.ColumnDataSource({
                     "tweet": _raw_tweet[:num_example],
                     "topic_key": _lda_keys[:num_example]
                   }))'''                

  # plot crucial words
  for i in range(X_topics.shape[1]):
    plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
  hover = plot_lda.select(dict(type=HoverTool))
  hover.tooltips = {"tweet": "@tweet - topic: @topic_key"}

  save(plot_lda, 'tsne_lda_viz_{}_{}_{}_{}_{}_{}.html'.format(
    num_qualified_tweet, n_topics, threshold, n_iter, num_example, n_top_words))


  t4 = time.time()
  print('\n>>> whole process done; took {} mins\n'.format((t4-t0)/60.))
