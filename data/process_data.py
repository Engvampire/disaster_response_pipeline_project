import sys
import pandas as pd
import sqlite3
from flask_sqlalchemy import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
	load_data :
        load messages & categories from .csv files and and merges them into one dataframe
	Input 
        messages_filepath : messages file path
        categories_filepath : categories file path
	Output :- 
        df : merged data dataframe
	"""
    messages = pd.read_csv(messages_filepath)
    messages = messages.fillna(0)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer',on='id')
    return df

def clean_data(df):
    """
	clean_data :
        clean the dataframe and change the form of the categories so they can be used in further analysis
	Input 
        df : data which needs to be cleaned dataframe
    Output :- 
        df : cleaned data dataframe
	"""
    categories = df.categories.str.split(pat=";", n=-1, expand=True)
    categories.columns = categories.iloc[0].str.split('-').str[0]

    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])
        categories.replace(2, 1, inplace=True)
    
    df1=df.drop(['categories'], axis=1)
    df2 = [df1, categories]
    df = pd.concat(df2, axis=1,sort=False)
    df=df.drop_duplicates(subset='message')
    return df

def save_data(df, database_filename):
    """
	save_data :
        Saving dataframe to a database
	Input 
        df : The dataframe to be saved.
        database_filename : the database file path of where the data will be saved.
    Output :- 
        NA
    """
    url="sqlite:///"+database_filename
    engine = create_engine(url)
    df.to_sql('msgs', engine, index=False,if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()