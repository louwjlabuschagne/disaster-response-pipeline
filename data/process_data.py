import sys
import os
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    # read in files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.values[0].split('-')[0]).values
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df.original.fillna('Already English', inplace=True)

    return df


def save_data(df, database_filename, db_name='Message', verbose=True):
    engine = create_engine('sqlite:///' + database_filename)
    with engine.begin() as conn:
        conn.execute('DROP TABLE IF EXISTS %s' % db_name)
        if verbose:
            print('Dropped %s table in %s database' %
                  (db_name, database_filename))
        for start in range(0, df.shape[0], 20):
            if start % 5000 == 0:
                if verbose:
                    print('saved %s records' % start)
            end = start + 20
            if start + 20 > df.shape[0]:
                end = df.shape[0]
            df[start:end].to_sql(name=db_name, con=conn,
                                 index=True, if_exists='append')
    if verbose:
        print('Wrote %d records to %s in %s' %
              (df.shape[0], db_name, os.path.abspath(database_filename)))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.shape)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
