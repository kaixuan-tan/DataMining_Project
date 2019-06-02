from pprint import pprint
import re
import pandas as pd
import numpy as np
#import nltk
#import matplotlib.pyplot as plt
#import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
#sns.set(style='darkgrid', context='talk', palette='Dark2')
#import csv

stockprice = pd.read_csv(r'...\stocknewsheadline.csv')
headlines = stockprice['Headline']
print(headlines)

sia = SIA()
results = []

def normalize_text(s):
    s = s.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)

    return s

def convertTuple(tup):
    str =  ''.join(tup)
    return str


#headlineText = [normalize_text(convertTuple(s)) in headlines]

for line in headlines:
    headlineText = normalize_text(convertTuple(line))
    pol_score = sia.polarity_scores(headlineText)
    pol_score['headline'] = headlineText
    results.append(pol_score)


#pprint(results[:5], width=100)

df = pd.DataFrame.from_records(results)

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1

print(df)
df.to_csv(r'...\headlines_labels.csv', encoding='utf-8', index=False)
dfmerged = stockprice.join(df)
dfmerged.to_csv(r'...\output.csv', encoding='utf-8', index=False)