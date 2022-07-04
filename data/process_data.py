import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
        1) load the messages and categories data 
        2) merge messages and categories on id column and return the merged dataframe
        
        Args:
            messages_filepath: filepath of the messages data in .csv format
            categories_filepath: filepath of the categories data in .csv format
        Returns:        
            df: merged data as a pandas dataframe
    '''

    # load messages data
    messages = pd.read_csv(messages_filepath)
    # load categories data
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
        Cleans the input dataframe and returns the dataframe:
        1) create a new categories dataframe
        2) extract and assign attribute (column) names
        3) extract and assign observation values (int)
        4) drop the original category column and concat the new categories
        5) remove duplicate data (original category and messages data had 68 duplicate rows)

        Args: 
            df: pandas dataframe containing merged raw data
        Returns: 
            df: pandas dataframe cleaned data
    '''
    
    # step 1
    # create a dataframe with individual category columns
    categories = df.categories.str.split(';',expand=True)
    
    # step 2
    # extract column names for categories.
    row = categories.iloc[0]
    category_col_names = row.astype(str).str[:-2]
    # rename/assign column names for categories
    categories.columns = category_col_names
    
    # step 3
    # set category values as the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    
    # step 4
    # drop the original categories column from df
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # step 5
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
        save the df as a sqlite database
        Args:
            df: pandas dataframe
            database_filename: str
        Returns: None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster_Response_Table', engine, index=False)  


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