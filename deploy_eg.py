from transformers import BertTokenizer,BertForSequenceClassification
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import torch
import numpy as np
import pandas as pd
import contractions
import streamlit as st

from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-8_H-512_A-8')
model = BertForSequenceClassification.from_pretrained("/Users/home/Desktop/My Project/Twitter Sentiment Analysis/Model")

#Removing stop-words and converting words to lemma

stop_words = set(stopwords.words('english')) 

lemmatizer = WordNetLemmatizer() 

def stop_wrds_lemma_convert(sentence):
    tokens = [w for w in sentence.split() if not w in stop_words] #stopwords removal
    newString=''
    for i in tokens:                                                 
        newString=newString+lemmatizer.lemmatize(i)+' '    #converting words to lemma                               
    return newString.strip()

def predict(text):
    clean_text=text.lower()#Converting them to lowercase
    clean_text=re.sub(r'@[A-Za-z0-9]+','',clean_text)#Remove user mentions from the tweet
    expanded_words = []   
    for word in clean_text.split():
        expanded_words.append(contractions.fix(word)) 
    clean_text = ' '.join(expanded_words)
    clean_text=re.sub('[^\w\s]','',clean_text)#Removing Puntuation
    clean_text=re.sub(r'[0-9]+','',clean_text)
    clean_text=re.sub(r'http\S+','',clean_text)
    clean_text=re.sub('#','',clean_text)#Remove hashtags from the tweet
    clean_text=stop_wrds_lemma_convert(clean_text)
    inputs = tokenizer(clean_text,padding = True, truncation = True, return_tensors='pt',max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    if(np.argmax(predictions) == 0):
        return "Negative"
    elif (np.argmax(predictions) == 1):
        return "Positive"
    else:
        return "Neutral"

# text = "He's Good Boy @!#"
# prediction = predict(text)
# print(prediction)
prediction = ''


st.title("😍 Twitter Sentiment Analysis 🦄")
text = st.text_input("Enter Text To Test 👀")

if st.button("Sentiment Result 🐼"):
    prediction = predict(text)

st.success(prediction)

