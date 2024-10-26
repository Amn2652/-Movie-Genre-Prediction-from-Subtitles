# -Movie-Genre-Prediction-from-Subtitles
CONTENT OF README 

# Movie Genre Prediction from Subtitles
# Overview
This project predicts movie genres based on subtitle files using natural language processing techniques. The model preprocesses the subtitle text, trains a logistic regression classifier using word embeddings or TF-IDF vectorization, and then predicts genre probabilities for the provided subtitle files.
# Features
- Advanced text preprocessing (tokenization, stemming, stopword removal)
- Support for Word2Vec and TF-IDF vectorization methods
- Logistic regression model for genre classification
- Cross-validation for model evaluation
- Genre prediction from subtitle files (.srt)
# Dataset
This project utilizes the dataset consists of:  
- CSV File: The dataset is structured in a CSV format with the following columns:
  - description: A textual representation of the movie's script or dialogue.
  - genre: The corresponding genre(s) of the movie (e.g., Action, Comedy, Drama,etc).
And many more columns such as director,gross etc.
Make sure to have the following Python packages installed:
- nltk
- numpy
- pandas
- gensim
- scikit-learn

You can install the required packages using pip:

```bash
pip install nltk numpy pandas gensim scikit-learn

## I have executed the code on google collab and it took around 5-10 mins to execute and there are 16 csv files and we have to provide path of .srt file. Accuracy approx. 30-35%.
