# -*- coding: utf-8 -*-
"""
Created on Wed May 29 05:37:36 2024

@author: Samson
"""

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TweetProcessor:
    def __init__(self):
        # Define the emoji pattern
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        self.punctuations = string.punctuation
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, data):
        # Ensure all values in the 'tweet' column are strings and handle missing values
        data['tweet'] = data['tweet'].astype(str).fillna('')

        data['tweet'] = data['tweet'].str.replace('@USER', '', regex=False)  # Remove mentions (@USER)
        data['tweet'] = data['tweet'].str.replace('URL', '', regex=False)  # Remove URLs
        data['tweet'] = data['tweet'].str.replace('&amp', 'and', regex=False)  # Replace ampersand (&) with 'and'
        data['tweet'] = data['tweet'].str.replace('&lt', '', regex=False)  # Remove &lt
        data['tweet'] = data['tweet'].str.replace('&gt', '', regex=False)  # Remove &gt
        data['tweet'] = data['tweet'].str.replace(r'\d+', '', regex=True)  # Remove numbers
        data['tweet'] = data['tweet'].str.lower()  # Convert to lowercase
        data['tweet'] = data['tweet'].str.replace(r'\buser\b', '', regex=True)  # Remove the word 'user'

        for punctuation in self.punctuations:
            data['tweet'] = data['tweet'].str.replace(punctuation, '', regex=False)

        data['tweet'] = data['tweet'].apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii')
        )  # Remove emojis
        data['tweet'] = data['tweet'].str.strip()  # Trim leading and trailing whitespaces

        data['cleaned_tweet'] = data['tweet'].apply(self.tokenize_and_lemmatize)
        return data

    def find_special_signs(self, text):
        # Remove emojis using the emoji pattern
        return re.sub(self.emoji_pattern, '', text)

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [word for word in tokens if word not in self.stop_words]  # Remove stop words
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the tokens
        return ' '.join(tokens)  # Join tokens into a single string

    def process(self, data):
        # Clean the 'tweet' column
        data = self.clean(data)
        # Apply find_special_signs to each tweet
        data['cleaned_tweet'] = data['cleaned_tweet'].apply(self.find_special_signs)
        # Tokenize and lemmatize again if needed
        data['cleaned_tweet'] = data['cleaned_tweet'].apply(self.tokenize_and_lemmatize)
        return data

