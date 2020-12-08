# import
import pandas as pd
import sys
import glob
import dask.dataframe as dd
import matplotlib.pyplot as plt
from utils import get_spain_places
import re

# args
raw_tweet_dir = sys.argv[1] # data path
scope = sys.argv[2] # SPA

# read files
# tweets
all_files = glob.glob(raw_tweet_dir + "/ours_*.csv")
raw_tweet_files = dd.read_csv(all_files,usecols=['tweet_id','date','user_id'],dtype={'user_id': 'str'})
raw_tweet_files = raw_tweet_files.compute()
# lang
all_files = glob.glob(raw_tweet_dir + "/tweets_*.csv")
raw_lang_files = dd.read_csv(all_files,usecols=['tweet_id','lang'])
raw_lang_files = raw_lang_files.compute()
# merge files 
data = pd.merge(raw_tweet_files,raw_lang_files, on='tweet_id')

# SPA
if scope=='SPA':
    # tweets users 
    all_files = glob.glob(raw_tweet_dir + '/users_loc*.csv')
    raw_user_file = dd.read_csv(all_files,usecols=['id_str','location'],dtype={'id_str': 'str'})
    raw_user_file = raw_user_file.compute()
    raw_user_file.rename({'id_str': 'user_id'}, axis=1, inplace=True)
    #
    data = data.merge(raw_user_file, on='user_id')
    # exclude latam but check spain places > 50k
    locations2check = "|".join(get_spain_places(raw_tweet_dir+"/places_spain.csv")) # get lists of spain places > 50k 
    data = data[data.apply(lambda x: len([s for s in str(x['location']).split() if re.compile(locations2check).match(s.lower())]) > 0, axis=1)]
    counting = set(data.loc[data['lang'].str.contains(lang)]['tweet_id'])
    print("SPA excluding locations tweets, es",len(counting))

# get stats
data['date'] = pd.to_datetime(data['date'])
# only by date
dataC = data.groupby([data.date.dt.year,data.date.dt.month]).count()
# write stats
with open("out/statistics-date{}.tsv".format(scope),'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))
# ... and by lang
dataC = data.groupby([data.date.dt.year,data.date.dt.month,data.lang]).count()
# write stats
with open("out/statistics-date-lang{}.tsv".format(scope),'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))
# ... and by lang gn
dataC = data[(data['lang'].str.contains('gn',regex=True))].groupby([data.date.dt.year,data.date.dt.month,data.lang]).count()
# write stats
with open("out/statistics-date-gn{}.tsv".format(scope),'w') as write_tsv:
    write_tsv.write(dataC[['date']].to_csv(sep='\t', encoding='utf-8'))

