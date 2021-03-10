#!/usr/bin/env python
# coding: utf-8

# # Dependencies

# In[1]:


# !pip install PyPDF2
# !pip install tabula
# !pip install textract
import nltk
import pandas as pd
import PyPDF2
import tabula
import textract
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[5]:


#open_filename_WS = open("/content/Witness Statement Pack.pdf", 'rb')
#open_filename_DB = open(r"/content/Digital Bundle.pdf", 'rb')
#open_filename_ET = open("/content/ET unmarked.pdf", 'rb')

open_filename_WS = open("C:/Users/yashd/OneDrive/Documents/Github Master/witness/New Project Files/Witness Statement Pack2.pdf", 'rb')
open_filename_DB = open("C:/Users/yashd/OneDrive/Documents/Github Master/witness/New Project Files/Digital Bundle1.pdf", 'rb')
open_filename_ET = open("C:/Users/yashd/OneDrive/Documents/Github Master/witness/New Project Files/ET unmarked3.pdf", 'rb')

WS = PyPDF2.PdfFileReader(open_filename_WS)

#DB = PyPDF2.PdfFileReader(open_filename_DB)
#ET = PyPDF2.PdfFileReader(open_filename_ET)
#print (open_filename)


# In[6]:


WS.getDocumentInfo()
#DB.getDocumentInfo()
#ET.getDocumentInfo()


# In[7]:


total_pages_WS = WS.numPages
print(total_pages_WS)

#total_pages_DB = DB.numPages
#print(total_pages_DB)

#total_pages_ET = ET.numPages
#print(total_pages_ET)


# # Datasets TEXT format

# In[8]:


count_WS = 0
text_WS  = ''

# Lets loop through, to read each page from the pdf file
while(count_WS < total_pages_WS):
    # Get the specified number of pages in the document
    mani_pageWS  = WS.getPage(count_WS)
    # Process the next page
    count_WS += 1
    # Extract the text from the page
    text_WS += mani_pageWS.extractText()
print(text_WS)


# In[9]:


nltk.download('punkt')


# In[10]:


sentencesWS = nltk.sent_tokenize(text_WS)
#sentencesDB = nltk.sent_tokenize(text_DB)
#sentencesET = nltk.sent_tokenize(text_ET)


# In[11]:


print(sentencesWS) 


# In[12]:


corpusWS = []


# In[13]:


import re
for i in range(len(sentencesWS)):
    reviewWS = re.sub('[^0-9a-z^A-Z]',' ',sentencesWS[i])
    reviewWS = reviewWS.lower()
    reviewWS = reviewWS.split()
    reviewWS = ' '.join(reviewWS)
    corpusWS.append(reviewWS)
corpusWS


# In[14]:


len(corpusWS)


# In[15]:


corpusWS


# In[25]:


sentenceWS=["sentenceWS"]

for x in corpusWS:
    sentenceWS.append(x)
    #print("Transcript: {}".format(result.alternatives[0].transcript))



with open("sentenceWSvendor.csv", 'w') as myfile:
    for x in sentenceWS:
        myfile.write(x)
        myfile.write("\n")


# In[21]:


#Sentiment Analysis of the PDF
#This is a set of Natural Language Processing (NLP) technique of analysing, identifying and categorizing opinions expressed in a piece of text, in order to determine whether the writer's attitude towards a particular topic, product, politics, services, brands etc. is positive, negative, or neutral.

# !pip install vaderSentiment
get_ipython().system('pip install textblob')
import vaderSentiment
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# #Lets decide which model we should use, between TextBlob and VADER for analysis of our text. We will therefore use TextBlob for its simplcity, and since VADER is specifically for analysis of social media data.
# #TextBlob function - returns two properties
# #Polarity: a float value which ranges from [-1.0 to 1.0] where 0 indicates neutral, +1 indicates most positive statement and -1 rindicates most negative statement.
# #Subjectivity: a float value which ranges from [0.0 to 1.0] where 0.0 is most objective while 1.0 is most subjective. Subjective sentence expresses some personal opinios, views, beliefs, emotions, allegations, desires, beliefs, suspicions, and speculations where as objective refers to factual information.

# In[22]:


import pandas as pd

# To read the CSV file
df = pd.read_csv('sentenceWSvendor.csv')

from textblob import TextBlob

# The x in the lambda function is a row (because I set axis=1)
# Apply iterates the function accross the dataframe's rows
df['polarity'] = df.apply(lambda x: TextBlob(x['sentenceWS']).sentiment.polarity, axis=1)
df['subjectivity'] = df.apply(lambda x: TextBlob(x['sentenceWS']).sentiment.subjectivity, axis=1)

print(df)


# In[23]:


get_ipython().system('pip install vadersentiment')
get_ipython().system('pip install nltk')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[24]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[60]:


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {} {} {} {}".format(sentence, str(score['neg']), str(score['neu']), str(score['pos']), str(score['compound'])))


# In[61]:


sentiment_analyzer_scores("The phone is super cool.")


# In[26]:


# To read the CSV file
df = pd.read_csv('sentenceWSvendor.csv')

df.head()


df['Negative'] = df.apply(lambda x: str(analyser.polarity_scores(x['sentenceWS'])['neg']), axis=1)
df['Neutral']  = df.apply(lambda x: str(analyser.polarity_scores(x['sentenceWS'])['neu']), axis=1)
df['Positive'] = df.apply(lambda x: str(analyser.polarity_scores(x['sentenceWS'])['pos']), axis=1)
df['Compound'] = df.apply(lambda x: str(analyser.polarity_scores(x['sentenceWS'])['compound']), axis=1)


# In[29]:


df


# In[30]:


df.to_csv('sentenceWSvendor.csv')

