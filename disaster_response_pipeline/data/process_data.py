import sys, os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    ::describe: load data from csv file
    ::param:
        messages_filepath: path to messages data
        categories_filepath: path to categories data
    ::return: merged data frame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    ::describe: clean data from input what data frame
    ::param:
        df: merged data frame
    ::return: cleaned data frame
    """
    cats = df["categories"].str.split(";", expand=True)
    cats.columns = cats.iloc[0, :].apply(lambda x: x.split("-")[0])
    for col in cats:
        cats[col] = cats[col].apply(lambda x: x.split("-")[1])
        cats[col] = pd.to_numeric(cats[col])
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, cats], axis=1)
    if df.duplicated().sum() > 0: df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    ::describe: Store cleaned data into Database
    ::param:
        df: cleaned data frame
        database_filename: name of db file
    ::return: None
    """
    db_engine = create_engine("sqlite:///" + database_filename)
    db_name = os.path.basename(database_filename).split(".")[0]
    try:
        df.to_sql(db_name, db_engine, index=False)
    except ValueError as error:
        print(error)
    

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