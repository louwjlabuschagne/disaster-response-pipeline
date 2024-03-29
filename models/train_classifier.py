import pickle as pck
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import sys
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Function that reads in data from a SQLite database

    Arguments
    database_filename :: str :: location of SQLite database

    Returns
    X :: pandas.DataFrame with all the messages
    Y :: pandas.DataFrame with the binary indication of whether the associated message falls into that category
    Columns :: pandas.Series with the categories for Y
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    with engine.begin() as conn:
        df = pd.read_sql('SELECT * FROM Message', con=conn)
    X = df.message
    Y = df.drop(['index', 'id', 'message',
                 'original', 'genre', 'index'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """
    Function that tokenizes provided text

    Arguments
    text :: str :: piece of text to tokenize

    Returns
    clean_tokens :: list of the clean tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(n_jobs=-1, low_cpu=True):
    """
    Function that constructs a sklearn model

    Arguments
    low_cpu :: boolean :: Defaults to True, but the function will create a much more thorough gridsearch if False, at the cost of CPU power.
    n_jobs :: number of CPUs to use - defaults to -1

    Returns
    cv :: model.selection.GridSearchCV that contains a sklearn pipeline with
        CountVectorizer(tokenizer=tokenize)
        TfidfTransformer()
        MultiOutputClassifier(RandomForestClassifier())

        which has been gridsearch with parameters defined within this function

    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    if low_cpu:
        parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                      'clf__estimator__min_samples_split': [4]}
    else:
        parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                      'vect__max_df': (0.5, 0.75, 1.0),
                      'vect__max_features': (None, 5000, 10000),
                      'tfidf__use_idf': (True, False),
                      'clf__estimator__n_estimators': [50, 100, 200],
                      'clf__estimator__min_samples_split': [4]}

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function that returns accuracy metrics for a given model

    Arguments
    model :: sklearn.Estimator to be used for prediction
    X_test :: pandas.DataFrame with messages to use for prediction
    Y_test :: pandas.DataFrame with the correct labels
    category_names :: list with the label names

    Returns
    resutls :: pandas.DataFrame with macro precision, accuracy, recall and f1 score for the model

    """
    Y_pred = model.predict(X_test)

    categories = []
    precisions = []
    accuracies = []
    recalls = []
    f1_scores = []

    for category, y_test, y_pred in zip(Y_test.columns, Y_test.values.T, Y_pred.T):
        categories.append(category)
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    results = pd.DataFrame(dict(category=categories,
                                precision=precisions,
                                accuracy=accuracies,
                                recall=recalls,
                                f1_score=f1_scores)).melt(id_vars='category',
                                                          value_vars=[
                                                              'precision', 'accuracy', 'recall', 'f1_score'],
                                                          var_name='metric')
    return results


def save_model(model, model_filepath):
    """
    Function that saves the model in a pickle

    Arguments
    model :: sklearn.Estimator to save
    model_filepath :: str :: location to save model to

    Returns
    Nothing
    """
    pck.dump(model, open(model_filepath, 'wb+'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
