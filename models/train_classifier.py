import sys
import pandas as pd
import re
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# nltk.download()


def load_data(database_filepath):
    '''
        Args:
            database_filepath: file path for the sqlite data base
        Returns:
            X: features data
            Y: classes (multilabel)
            category_names: class names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_Response_Table', con=engine)
    
    category_names = df.columns[4:]
    X = df.message
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    '''
        This function performs a series of steps:
            a) tokenize text into sentences using sent_tokenize
            b) for each sentence, normalize the text 
            c) tekenize sentences into words using word_tokenize
            d) tag each word to indicate the part of speech (pos)
            e) join pos tag and the words
        Args:
            text: raw text
        Returns:
            joined_sent_words_pos: list of joined part of speach tags and words in different sentences 
    '''

    sent_words_pos=[]
    joined_sent_words_pos=[]
    sent_t = sent_tokenize(text) # tokenize the text into the sentences
    
#     for each sentence perform a series of processes
    for counter, sentences in enumerate(sent_t):
#       1  normalize text
        sentences = re.sub('[^a-zA-Z0-9_-]', ' ', sentences)
#       2  extract words
        words = word_tokenize(sentences)
#       3  remove stop words
        words = [w for w in words if w not in set(stopwords.words("english"))]
#       4  part of speach tag
        words_pos = pos_tag(words)
        
        sent_words_pos += words_pos # list of tuples, part of speach tags for each word in different sentences

        for item in sent_words_pos:
            joined_sent_words_pos.append(''.join(item)) # list: joined pos_tag and words tuple
        
    return joined_sent_words_pos


def build_model():
    '''
        RandomForestClassifier performs extremely well for fitting the training set, 
            the accuracy is greater than 99%.
        It's accuracy is in the order of 94% for the test set,
            and performs better than other classifiers listed below:

        Other classification algorithms such as SVC, MultinomialNB, Logistic Regression,
            and Adaboost were evaluated and RandomForest outperformed these algorithms by a margin.
    
        The modeling pipeline includes the following steps:
            - Count Vectorizer
            - tfidf transformer
            - Truncated SVD (optional), it may improve classification runtime but depreciates accuracy
            - Random forest classifier with Multioutput classifier

        Args: None            
        Returns: modeling pipeline            
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)), 
        ('tfidf', TfidfTransformer()),
        # ('t_svd', TruncatedSVD(n_components=100, n_iter=10, random_state=42)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))],
        verbose=True)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Evaluates the model against the test set and prints the classification report
        Args:
            model: machine learning model
            test data:
                        X_test: features data
                        Y_test: classes (multilabel)
                        category_names: class names
        Returns: None
            
    '''
    pred = model.predict(X_test)
    accuracy = (pred == Y_test.values).mean()
    print('The model accuracy is {:.2f}%'.format(accuracy*100))
    print('\n')

    for i, feat in enumerate(category_names):
        print('Feature {}: {}'.format(i + 1, feat))
        print(classification_report(Y_test.iloc[:, i], pred[:, i]))
        print('\n')


def save_model(model, model_filepath):
    '''
        Saves the machine learning model using pickle
        Args:
            model: machine learning model, that is trained and evaluated
            model_filepath: str
        Returns: None            
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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