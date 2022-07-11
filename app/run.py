import json
from turtle import title
from unicodedata import category
import plotly
import pandas as pd
import joblib
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine

from plotly.subplots import make_subplots


app = Flask(__name__)

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
    for _ , sentences in enumerate(sent_t):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Response_Table', con=engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data for figure_1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # figure_1
    figure_1 = Bar(x=genre_names, y=genre_counts)

    layout_1 = dict(title = 'Message Genre',
                xaxis = dict(title='', categoryorder='total descending'),
                yaxis = dict(title='Count')
                )
    
    # extract data for figure_2
    col_2_1 = ['related', 'request', 'offer', 'direct_report']    
    col_2_2 = df.columns[7:22]
    col_2_3 = df.columns[31:]
    col_2_4 = df.columns[23:31]

    # figure_2 - with 4 subplots 
    figure_2 = make_subplots(rows=1, cols=4,
                            subplot_titles=("General", "Aid", "Weather", "Infrastructure"),
                            column_widths=[4/35, 15/35, 7/35, 9/35], 
                            shared_yaxes=True, 
                            horizontal_spacing=0.005)
    # add 4 subplots    
    figure_2.add_trace(Bar(x=col_2_1, y=df[col_2_1].sum(axis=0), marker_color='steelblue'), row=1, col=1)
    figure_2.add_trace(Bar(x=col_2_2, y=df[col_2_2].sum(axis=0), marker_color='steelblue'), row=1, col=2)
    figure_2.add_trace(Bar(x=col_2_3, y=df[col_2_3].sum(axis=0), marker_color='steelblue'), row=1, col=3)
    figure_2.add_trace(Bar(x=col_2_4, y=df[col_2_4].sum(axis=0), marker_color='steelblue'), row=1, col=4)
    
    figure_2.update_xaxes(categoryorder='total descending', tickangle=45)
    figure_2.update_yaxes(showline=False, linewidth=0.1, linecolor='black', mirror=False)
    figure_2.update_yaxes(title='Count', row=1, col=1)

    figure_2.update_layout(showlegend=False, title={'text': "Message Categories",
                                                    'x': 0.5, 
                                                    'y': 0.9, 
                                                    'xanchor': "center",
                                                    'yanchor': "top"})


    figures=[]
    figures.append(dict(data=[figure_1], layout=layout_1))
    figures.append(dict(data=figure_2))
    

    # encode plotly graphs in JSON
    # ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == '__main__':
    app.debug = True
    # app.run(host="0.0.0.0", port=5000)