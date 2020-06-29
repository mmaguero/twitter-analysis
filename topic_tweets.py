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
import datetime
#

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
  #                   
  parser.add_argument('--start_date', required=True, type=str,
                      help='start date of data split, format: yyyy-mm-dd')                    
  parser.add_argument('--end_date', required=True, type=str,
                      help='end date of data split') 
  args = parser.parse_args()

  # unpack
  raw_tweet_dir = args.raw_tweet_dir
  num_train_tweet = args.num_train_tweet
  n_topics = args.n_topics
  n_iter = args.n_iter
  n_top_words = args.top_n
  threshold = args.threshold
  num_example = args.num_example
  #
  start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
  end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()


  ##############################################################################
  # get training tweets

  num_scanned_tweet = 0
  num_qualified_tweet = 0

  #raw_tweet_files = os.listdir(raw_tweet_dir)
  all_files = glob.glob(raw_tweet_dir + "/ours_*.csv")
  raw_tweet_files = dd.read_csv(all_files,usecols=['tweet_id','tweet','date'])
  raw_tweet_files = raw_tweet_files.compute()
  
  all_files = glob.glob(raw_tweet_dir + "/tweets_*.csv")
  raw_lang_files = dd.read_csv(all_files,usecols=['tweet_id','lang'])
  raw_lang_files = raw_lang_files.compute()
  
  # split by date
  raw_tweet_files['date'] = pd.to_datetime(raw_tweet_files['date']).dt.date
  raw_tweet_files = raw_tweet_files.loc[(raw_tweet_files['date'] <= end_date) & (raw_tweet_files['date'] >= start_date)]
  
  raw_tweet_merge = raw_tweet_files.merge(raw_lang_files, on='tweet_id')
  raw_tweet_merge.info()
  raw_tweet_text = set(raw_tweet_merge.loc[raw_tweet_merge['lang'].str.contains("es")]['tweet'])

  #raw_tweet_text = raw_tweet_text.reset_index(drop=True)
  #raw_tweet_utext = raw_tweet_text.drop_duplicates()
  #raw_tweet_utext = raw_tweet_utext.reset_index(drop=True)
  #raw_tweet_utext = set(raw_tweet_text['tweet'].tolist())
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

  # ignore terms that have a document frequency strictly lower than 1, 3, 5
  try:
      cvectorizer = CountVectorizer(min_df=5)
      cvz = cvectorizer.fit_transform(processed_tweet)
  except:
      try:
          cvectorizer = CountVectorizer(min_df=3)
          cvz = cvectorizer.fit_transform(processed_tweet)
      except:
          cvectorizer = CountVectorizer(min_df=1)
          cvz = cvectorizer.fit_transform(processed_tweet)

  lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
  X_topics = lda_model.fit_transform(cvz)

  t2 = time.time()
  print('\n>>> LDA training done; took {} mins\n'.format((t2-t1)/60.))
  
  try:
      np.save('lda_simple/lda_doc_topic_{}tweets_{}topics_{}.npy'.format(
    X_topics.shape[0], X_topics.shape[1], end_date), X_topics)
      np.save('lda_simple/lda_topic_word_{}tweets_{}topics_{}.npy'.format(
    X_topics.shape[0], X_topics.shape[1], end_date), lda_model.topic_word_)
      print('\n>>> doc_topic & topic word written to disk\n')
  except Exception as e:
      print('\n>>> doc_topic & topic word written to disk\n', e, '\n')
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
  topic_keywords = []
  topic_word = lda_model.topic_word_  # get the topic words
  vocab = cvectorizer.get_feature_names()
  for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(str(i) + ':' + ','.join(topic_words))
    topic_keywords.append(','.join(topic_words))
  
  #
  sent_topics_df = pd.DataFrame()
  # Get main topic in each document
  for i, topics in enumerate(X_topics):
      # Get the Dominant topic, Perc Contribution and Keywords for each document
      sent_topics_df = sent_topics_df.append(pd.Series([int(topics.argmax()), round(topics[topics.argmax()],4), topic_keywords[topics.argmax()]]), ignore_index=True)
  sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
  # Add original text to the end of the output
  contents = pd.Series(_raw_tweet)
  sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
  #

  # use the coordinate of a random tweet as string topic string coordinate
  topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
  _lda_keys = [int(k) for k in _lda_keys]
  for topic_num in _lda_keys:
    if not np.isnan(topic_coord).any():
      break
    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

  # plot

  title = "t-SNE visualization of LDA model trained on {} tweets, {} topics, " \
          "thresholding at {} topic probability, {} iter ({} data points and " \
          "top {} words: {})".format(num_qualified_tweet, n_topics, threshold,
                                 n_iter, num_example, n_top_words, end_date)

  plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                       title=title,
                       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)
  plot_lda.title.text_font_size = '8pt'
  plot_lda.axis.major_label_text_font_size="8pt"
  plot_lda.yaxis.axis_label_text_font_size = "8pt"
  plot_lda.xaxis.axis_label_text_font_size = "8pt"

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

  # plot crucial words
  for i in range(X_topics.shape[1]):
    #
    try:
      topic_coord[i, 0] = 0 if np.isnan(topic_coord[i, 0]) else topic_coord[i, 0]
      topic_coord[i, 1] = 0 if np.isnan(topic_coord[i, 1]) else topic_coord[i, 1]
      plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
    except:
      print("Error in plot_lda...",str(i)) 
  #   
  hover = plot_lda.select(dict(type=HoverTool))
  hover.tooltips = {"tweet": "@tweet - topic: @topic_key"}

  name = 'tsne_lda_viz_{}_{}_{}_{}_{}_{}_{}.html'.format(
    num_qualified_tweet, n_topics, threshold, n_iter, num_example, n_top_words, end_date)
  save(plot_lda, name, title=name.replace(".html",""))
  
  #
  # Group top 5 sentences under each topic
  sent_topics_sorteddf = pd.DataFrame()
  sent_topics_outdf_grpd = sent_topics_df.groupby('Dominant_Topic')
  for i, grp in sent_topics_outdf_grpd:
    print(grp)
    sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(5)], 
                                            axis=0)
  # Reset Index    
  sent_topics_sorteddf.reset_index(drop=True, inplace=True)
  # Format
  sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
  # Show
  sent_topics_sorteddf.sample(5)
  # save to disk
  sent_topics_sorteddf.to_csv(name.replace(".html",".tsv"), sep='\t', encoding='utf-8', index=False)
  #

  t4 = time.time()
  print('\n>>> whole process done; took {} mins\n'.format((t4-t0)/60.))
