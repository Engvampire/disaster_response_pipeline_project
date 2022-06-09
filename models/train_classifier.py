import sys
from sqlalchemy import create_engine
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """
	load_data :
        load data from database and extracts the messages, categories and category names
	Input 
        database_filepath : database file path
	Output :- 
        X : messages dataframe
        Y : categories dataframe
        category_names : a list of category names
	"""
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM msgs", engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    """
	tokenize :
        convert the plan text to a vey basic form which can be used as input to ML models
	Input 
        text : plan text dataframe
	Output :- 
        text : plan text dataframe
        clean_tokens : tekonized text dataframe
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
	build_model :
        Building a GridSearchCV model using a pipeline of 
        CountVectorizer,TfidfTransformer & MultiOutputClassifier-RandomForestClassifier.
	Input 
        NA
	Output :- 
        model : The GridSearchCV model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [5],
    
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=3, cv=4)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
	evaluate_model :
        Finsing F1 score, precision and recall for the test set is outputted for each category.
	Input 
        model : The model to be evaluated.
        X_test : The testing messages
        Y_test : The testing categories
        category_names : the category names (needed for printing purposes)
	Output :- 
        Results are printed to user
    """
    Y_pred = model.predict(X_test)
    
    Y_test_df = pd.DataFrame(Y_test, columns = category_names)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    for col in Y_test_df.columns:
        print("category: ", col,"========================================")
        print(classification_report(np.hstack(Y_test_df[col]), np.hstack(Y_pred_df[col])))

        
def save_model(model, model_filepath):
    """
	save_model :
        Saving model in the form of .pkl file
	Input 
        model : The model to be saved.
        model_filepath : the file path of where the model will be saved.
    Output :- 
        NA
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
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