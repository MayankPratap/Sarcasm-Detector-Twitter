import numpy as np  
import pandas as pd 
from sklearn import svm 
from sklearn.metrics import accuracy_score
import random

from wordsegment import segment
from nltk import word_tokenize,pos_tag
import enchant
import nltk
import string
import csv

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re

from enchant.tokenize import get_tokenizer,HTMLChunker

df=pd.read_csv('train.csv')

tweets=[]
tests=[]
labels_train=[]
features_train=[]
features_test=[]
labels_test=[]
punctuations=['.','"','{','(','-','!','?',':']

def removePunctuations(text):
    #sentence_list=nltk.sent_tokenize(text)
    #num_sentences=len(sentence_list)
    
    big_regex=re.compile('|'.join(map(re.escape,punctuations)))
    text=big_regex.sub(" ",text)

    text=' '.join(text.split())

    return text

def removeNumbers(text):
    text=re.sub(r'\d+','',text)
    return text

def removeHashTags(text):
    text=re.sub('#([^\\s]*)','',text)
    return text

cnt=0

DictTweet={}
DictLabel={}

for index,row in df.iterrows():
    cnt=cnt+1   
    #print(row['tweet'])
    #tweets.append(row['tweet'])
    DictTweet[row['ID']]=row['tweet']
    DictLabel[row['ID']]=row['label']
    #if row['label']=='sarcastic':
     #  labels_train.append(1)
    #else:
    #   labels_train.append(0)
    #labels_train.append(row['label'])

keys=list(DictTweet.keys())
random.shuffle(keys)

for key in keys:
    tweets.append(DictTweet[key])
    if DictLabel[key]=='sarcastic':
        labels_train.append(1)
    else:
        labels_train.append(0)




for i in range(0,len(tweets)):
    #print(tweets[i])
    # Removing b' or b" in front of tweet
    tweets[i]=tweets[i][2:]
    #print(tweets[i])
    if tweets[i][len(tweets[i])-1]=="'" or tweets[i][len(tweets[i])-1]=='"':
        tweets[i]=tweets[i][:len(tweets[i])-1]
    
    #print(tweets[i])

    tweets[i]=removePunctuations(tweets[i])
    tweets[i]=removeNumbers(tweets[i])
    tweets[i]=removeHashTags(tweets[i])
    tweets[i]=' '.join(tweets[i].split(' '))



intensifiers=[
                "amazingly",
                "-ass",
                "astoundingly",
                "awful",
                "bare",
                "bloody",
                "crazy",
                "dead",
                "dreadfully",
                "colossally",
                "especially",
                "exceptionally",
                "excessively"
                "extremely"
                "extraordinarily"
                "fantastically"
                "frightfully"
                "fucking"
                "fully"
                "hella"
                "holy"
                "incredibly"
                "insanely"
                "literally"
                "mad"
                "mightily"
                "moderately"
                "most"
                "outrageously"
                "phenomenally"
                "precious"
                "quite"
                "radically"
                "rather"
                "real"
                "really"
                "remarkably"
                "right"
                "sick"
                "so"
                "somewhat"
                "strikingly"
                "super"
                "supremely"
                "surpassingly"
                "terribly"
                "terrifically"
                "too"
                "totally"
                "uncommonly"
                "unusually"
                "veritable"
                "very"
                "wicked"
             ]

def custom_word_tokenize(text):
    tokenizer=get_tokenizer("en_US")
    words=[]

    for w in tokenizer(text):
        words.append(w[0])

    return words

def containsIntensifiers(text):

    check=0
    words_custom=custom_word_tokenize(text)

    for word in words_custom:
        for i in intensifiers:
            if i in word:
                check=1
                break

        if check==1:
            break

    return check

from textblob import TextBlob

def findTweetSentiment(text):
    blob=TextBlob(text)
    sum=0.0
    cnt=0
    for sentence in blob.sentences:
        sum+=sentence.sentiment.polarity
        cnt+=1

    try:
        return sum*1.0/cnt
    except ZeroDivisionError:
        return 0

def getWordSentiment(word):
    blob=TextBlob(word)
    return blob.sentiment.polarity

def getCapCount(text):
    words_custom=custom_word_tokenize(text)
    cnt1=0
    cnt2=0
    for word in words_custom:
        if word[0].isupper():
            cnt1+=1
        if word.isupper():
            cnt2+=1

    return cnt1,cnt2

def getCounts(text):
    words_custom=custom_word_tokenize(text)
    pos_tags=nltk.pos_tag(words_custom)

    tuples=pos_tags
    noun_count=0
    verb_count=0
    adj_count=0
    adverb_count=0

    for t in tuples:
        if 'NN' in t[1]:
            noun_count+=1
        elif 'JJ' in t[1]:
            adj_count+=1
        elif 'VB' in t[1]:
            verb_count+=1
        elif 'RB' in t[1]:
            adverb_count+=1

    return noun_count,verb_count,adj_count,adverb_count

import math

def get_all_features(text):
    features=[]
    
    #Has an intensifier
    check=containsIntensifiers(text)
    #print(check)
    if math.isnan(check) or check>=float('inf') or check<=float("-inf"):
        features.append(0)
    else:
        features.append(check)

    #sentiment of tweet
    senti=findTweetSentiment(text)
    #print(senti)
    if math.isnan(senti) or senti>=float('inf') or senti<=float("-inf"):
        features.append(0)
    else:
        features.append(senti)
    
    words_custom=custom_word_tokenize(text)
    #maximum word sentiment score
    maxi=float('inf')
    
    for word in words_custom:
        wordsenti=getWordSentiment(word)
        if wordsenti>maxi:
            maxi=wordsenti

    #print(maxi)
    # minimum word sentiment score
    mini=float('-inf')
 
    for word in words_custom:
        wordsenti=getWordSentiment(word)
        if wordsenti<mini:
            mini=wordsenti
    #print(mini)

    # diff between maximum and minimum sentiment scores
    diff=maxi-mini
    #print(diff)
    if math.isnan(maxi) or maxi>=float('inf') or maxi<=float("-inf"):
        features.append(0)
    else:
        features.append(maxi)

    if math.isnan(mini) or mini>=float('inf') or mini<=float("-inf"):
        features.append(0)
    else:
        features.append(mini)

    if math.isnan(mini) or diff>=float('inf') or diff<=float('-inf'):
        features.append(0)
    else:
        features.append(diff)

    # Number of words with initial caps

    cntInitCaps,cntAllCaps=getCapCount(text)
    #print(cntInitCaps)
    # Number of words with all caps
    #print(cntAllCaps)
    if math.isnan(cntInitCaps):
        features.append(0)
    else:
        features.append(cntInitCaps)
    if math.isnan(cntAllCaps):
        features.append(0)
    else:
        features.append(cntAllCaps)

    # Ratio of nouns to all words,verbs to all words,adjectives to all words,adverbs to all words
    cnt1,cnt2,cnt3,cnt4=getCounts(text)
    #print(cnt1)
    #print(cnt2)
    #print(cnt3)
    #print(cnt4)
    
    try:
        features.append(cnt1*1.0/len(words_custom))
        features.append(cnt2*1.0/len(words_custom))
        features.append(cnt3*1.0/len(words_custom))
        features.append(cnt4*1.0/len(words_custom))
    except ZeroDivisionError:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)


    return features

cnt=1
for tweet in tweets:
    #print(cnt)
    cnt+=1
    features_train.append(get_all_features(tweet))


l2=int(0.67*len(features_train))
for i in range(l2+1,len(features_train)):
    features_test.append(features_train[i])

features_train=features_train[:l2+1]

l2=int(0.67*len(labels_train))
for i in range(l2+1,len(labels_train)):
    labels_test.append(labels_train[i])

labels_train=labels_train[:l2+1]

print(len(features_train),len(features_test),len(labels_train),len(labels_test))


clf=svm.SVC()
clf.fit(features_train,labels_train)

pred=clf.predict(features_test)
score=accuracy_score(labels_test,pred)

print(score)






















