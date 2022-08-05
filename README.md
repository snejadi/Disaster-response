# Disaster Response Pipeline Project

## Summary
<br> The objective of this project is to categorize messages sent during disaster events.
  - The machine learning pipeline categorizes the events to send the messages to an appropriate disaster relief agency.
  - The web app displays the visualizations of the data and results.

## The Data
<br> The data set used in this study contains real messages sent during disaster events. It consists of more than 26,000 messages with 36 categories. The data is provided by [Appen](https://appen.com/) formally <i>Figure 8</i> to build a model for an API that classifies disaster messages.  
<br>
<br> [messages](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/data/disaster_messages.csv)
<br> [categories](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/data/disaster_categories.csv)
<br> 
<br> The following bar chart shows the distribution of data among different categories:
![Figure_01]()
<br> The following bar chart shows different message categories and the relative frequency distribution of text data belonging to each category:
![Figure_02]()

## Analysis
<br> This project involve three different componenets:

### 1- ETL Pipeline:
<br> The python script [process_data.py](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/data/process_data.py) is a data cleaning pipeline that:
  - Loads [messages](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/data/disaster_messages.csv) and [categories](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/data/disaster_categories.csv) datasets
  - Merges the datasets
  - Cleans the data
  - Stores cleaned data in a SQLite database

### 2- ML Pipeline:
<br> The python script [train_classifier.py](https://github.com/snejadi/Disaster-response/blob/e742fb0e32729cf6b2a5417f5937afc5bbd14a8c/models/train_classifier.py) is a machine learning pipeline that:
  - Loads the data from the SQLite database
  - Splits the dataset into traian and test sets
  - Builds the <i>text processing</i> and <i>machine learning</i> pipelines
  - Trains and tunes the model
  - Evaluates the model on the test set
  - Stores the final model as a pickle file

### 3- Web App:
<br> The Flask web app presents the visualizations and the classification results.

## Instructions:
1. Run the following commands in the project's root directory to set up the database and the model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory

3. Run the web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Licensing, Acknowledgements
<br> The author wishes to acknowledge [Udacity](https://www.udacity.com/) and [Appen](https://appen.com/) (Figure 8). The data, and templates are provided by Udacity.

## Author
<br>[Siavash Nejadi](https://github.com/snejadi/)