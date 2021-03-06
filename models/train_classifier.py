
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import time
import re
import pickle
import string


import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, make_scorer, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    '''
    Loads data from the cleaned database, tokenizes them, 
    builds machine learning model (with parameters found in RandomizedSearchCV),
    evaluates it and saves it as pickle file.
    
    Args:
        database_filepath (string) - path to database 
        
    Returns:
        X (numpy array) - array of data (without target variables)
        y (numpy array) - array of target variables
        category_names (list of strings) - full list of category names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessagesDatabase', engine)
    X = df.message.values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.tolist()
    return X, y, category_names

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

def tokenize(text):
    '''
    Tokenizes text, lemmatizes obtained list of tokens and cleans it from stoplist and symbols
    
    Args:
        text (string): text to tokenize
    
    Returns:
        clean_tokens (list): list of cleaned tokens
    '''
    tokens = TweetTokenizer().tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    clean_tokens = [tok for tok in clean_tokens if tok not in STOPLIST]
    clean_tokens = [tok for tok in clean_tokens if tok not in SYMBOLS]
    
    return clean_tokens


def build_model():
    '''
    Builds machine learning model (with parameters previously found in RandomizedSearchCV) 
    
    Args:
        None
    
    Returns:
        model - machine learning pipeline 
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, 
                                 stop_words='english', 
                                 max_df=0.5, 
                                 max_features=20000,
                                 min_df=5,
                                 ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultiOutputClassifier(MultinomialNB(alpha=0.02)))
    ])
        
    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Prints classification report for a given list of category_names
    using a given model evaluated on test dataset (X_test, y_test).
    
    Args:
        model - machine learning pipeline
        X_test (numpy array) - array of test data (without target variables)
        y_test (numpy array) - array of test target variables
        category_names (list of strings) - list of category names consistent with y_test
    
    Returns:
        None
    '''
    y_test_pred = model.predict(X_test)
    for i, item in enumerate(category_names):
        print('>>>', item)
        print(classification_report(y_test[:,i], y_test_pred[:,i]))

def save_model(model, model_filepath):
    '''
    Saves the selected model in pickle.
    
    Args:
        model - machine learning pipeline
        model_filepath (string) - path to save the model
    
    Returns:
        None
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Runs the abovementioned procedures in a pipeline 
    (build_model -> model.fit -> evaluate_model -> save_model).
    
    Args:
        None
        
    Returns:
        None
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()