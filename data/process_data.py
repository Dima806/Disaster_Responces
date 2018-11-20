
# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def extract(messages_str, categories_str):
    '''
    Extracts raw CSV data, prints shapes of corresponding dataframes and returns them.
    
    Args:
        messages_str (string) - a path to messages.csv file
        categories_str (string) - a path to categories.csv file

    Returns:
        messages (pandas dataframe) - dataframe with messages
        categories (pandas dataframe) - dataframe with categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_str)
    print('>>> messages shape: ', messages.shape)

    # load categories dataset
    categories = pd.read_csv(categories_str)
    print('>>> categories shape: ', categories.shape)

    return messages, categories

def transform(messages, categories):
    '''
    Merges and transforms extracted data (messages and categories).
    
    Args:
        messages (pandas dataframe) - dataframe with messages
        categories (pandas dataframe) - dataframe with categories        
    Returns:
        all_df (pandas dataframe) - transformed dataframe
    '''
    # merge datasets
    all_df = messages.merge(categories, how='outer', on='id')
    print('>>> all_df shape: ', all_df.shape)

    # create a dataframe of the 36 individual category columns
    cat_df = categories.categories.str.split(';', expand=True)
    print('>>> cat_df shape: ', cat_df.shape)

    # select the first row of the categories dataframe
    row = cat_df.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda s: s.split('-')[0]).tolist()
    print('>>> category colnames: ', category_colnames)

    # rename the columns of `categories`
    cat_df.columns = category_colnames

    # convert cat_df values to 0 and 1
    for column in cat_df:
        # set each value to be the last character of the string
        cat_df[column] = cat_df[column].apply(lambda s: s[-1])
        # convert column from string to numeric
        cat_df[column] = pd.to_numeric(cat_df[column])

    # drop the original categories column from `df`
    all_df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    all_df = pd.concat([all_df, cat_df], axis=1)
    
    # sort out 2s from `related` column
    all_df = all_df[all_df.related < 2.0]
        
    # convert all float64 columns to bool
    for col in all_df.select_dtypes(include=['float64']):
        all_df[col] = all_df[col].astype(bool)
    
    # check number of duplicates
    print('>>> number of duplicates before dropping: ', all_df.duplicated(subset=None, keep='first').sum())

    # drop duplicates
    all_df.drop_duplicates(subset=None, keep='first', inplace=True)

    # check number of duplicates
    print('>>> number of duplicates after dropping: ', all_df.duplicated(subset=None, keep='first').sum())

    # print the final shape
    print('>>> all_df shape: ', all_df.shape)
    
    return all_df

def load(all_df, database_str):
    '''
    Loads trasnformed data back to database.
    
    Args:
        all_df (pandas dataframe) - transformed dataframe
        database_str (string) - a path to final cleaned database
    Returns:
        None
    '''
    
    engine = create_engine('sqlite:///'+database_str)
    all_df.to_sql('DisasterMessagesDatabase', engine, index=False, if_exists='replace')

def make_etl(messages_str, categories_str, database_str):
    '''
    Extract-Transform-Load procedure for disaster messages and message categories.
    
    Args:
        messages_str (string) - a path to messages.csv file
        categories_str (string) - a path to categories.csv file
        database_str (string) - a path to final cleaned database

    Returns:
        None    
    '''
    
    messages, categories = extract(messages_str, categories_str)
    
    all_df = transform(messages, categories)
    
    load(all_df, database_str)
    
    print('>>> DONE')
    

if __name__ == '__main__':
    messages_str, categories_str, database_str = sys.argv[1:]
    print(messages_str, categories_str, database_str)
    make_etl(messages_str, categories_str, database_str)