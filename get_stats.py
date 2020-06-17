# import
import pandas as pd
import sys
import glob
import dask.dataframe as dd
import matplotlib.pyplot as plt

# args
raw_tweet_dir = sys.argv[1] # data path

# read files
# tweets
all_files = glob.glob(raw_tweet_dir + "/ours_*.csv")
raw_tweet_files = dd.read_csv(all_files,usecols=['tweet_id','date'])
raw_tweet_files = raw_tweet_files.compute()
# lang
all_files = glob.glob(raw_tweet_dir + "/tweets_*.csv")
raw_lang_files = dd.read_csv(all_files,usecols=['tweet_id','lang'])
raw_lang_files = raw_lang_files.compute()
# merge files 
data = pd.merge(raw_tweet_files,raw_lang_files, on='tweet_id')

# get stats
data['date'] = pd.to_datetime(data['date'])
# only by date
dataC = data.groupby([data.date.dt.year,data.date.dt.month]).count()
# write stats
with open("statistics-date.tsv",'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))
# ... and by lang
dataC = data.groupby([data.date.dt.year,data.date.dt.month,data.lang]).count()
# write stats
with open("statistics-date-lang.tsv",'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))
# ... and by lang gn
dataC = data[(data['lang'].str.contains('gn',regex=True))].groupby([data.date.dt.year,data.date.dt.month,data.lang]).count()
# write stats
with open("statistics-date-gn.tsv",'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))
