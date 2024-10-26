import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Set environment variable to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download nltk data for stopwords and stemming
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

# Function for advanced text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return filtered_tokens

# Load CSV files into DataFrames
def load_movie_script_dataframes(directory='.'):
    dataframes = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            genre = file_name.split('.')[0]  # Assuming genre is part of the filename
            df = pd.read_csv(os.path.join(directory, file_name))
            dataframes[genre] = df
    return dataframes

# Load movie script dataset from DataFrames
def load_movie_script_dataset(dataframes):
    scripts = []
    genres = []
    for genre, df in dataframes.items():
        scripts.extend(df['description'].tolist())  # Use the 'description' column for text
        genres.extend([genre] * len(df))  # Add genre labels
    return scripts, genres

# Train Word2Vec model
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Vectorize with Word2Vec using weighted average
def vectorize_with_word2vec(sentences, model):
    vectors = []
    for sentence in sentences:
        words = [word for word in sentence if word in model.wv.key_to_index]
        if words:
            weights = np.array([model.wv[word] for word in words])
            tfidf_weights = np.array([1] * len(words))  # Uniform weights
            vectors.append(np.average(weights, axis=0, weights=tfidf_weights))
        else:
            vectors.append(np.zeros(model.vector_size))  # Zero vector for empty sentences
    return np.array(vectors)

# Vectorize with TF-IDF
def vectorize_with_tfidf(scripts):
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)  # Remove rare words
    return vectorizer.fit_transform(scripts), vectorizer

# Main function to train the model and predict genre probabilities from an .srt file
def train_and_predict_genre(srt_path, dataframes, use_word2vec=True):
    # Step 1: Extract text and preprocess from the .srt file
    script_text = extract_text_from_srt(srt_path)
    cleaned_text = preprocess_text(script_text)

    # Step 2: Load dataset and preprocess from DataFrames
    movie_scripts, movie_genres = load_movie_script_dataset(dataframes)
    preprocessed_scripts = [preprocess_text(script) for script in movie_scripts]

    # Step 3: Choose method for vectorization
    if use_word2vec:
        word2vec_model = train_word2vec(preprocessed_scripts)
        X = vectorize_with_word2vec(preprocessed_scripts, word2vec_model)
    else:
        X, vectorizer = vectorize_with_tfidf(movie_scripts)

    # Step 4: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, movie_genres, test_size=0.2, random_state=42)

    # Step 5: Train Logistic Regression model with cross-validation
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'Cross-Validation Accuracy: {np.mean(cross_val_scores) * 100:.2f}%')

    model.fit(X_train, y_train)

    # Step 6: Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

    # Step 7: Predict genres for the .srt file
    if use_word2vec:
        cleaned_script_vector = vectorize_with_word2vec([cleaned_text], word2vec_model)
    else:
        cleaned_script_vector = vectorizer.transform([' '.join(cleaned_text)])

    genre_probabilities = model.predict_proba(cleaned_script_vector)[0]

    # Return genre ratings without applying threshold
    genre_ratings = {model.classes_[i]: round(genre_probabilities[i] * 100, 2) for i in range(len(genre_probabilities))}

    return genre_ratings

# Function to extract text from .srt files
def extract_text_from_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    subtitle_text = ''
    for line in lines:
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3}', line) or line.strip() == '':
            continue
        subtitle_text += line.strip() + ' '

    return subtitle_text

# Load the uploaded CSV files from the Files section
dataframes = load_movie_script_dataframes()  # Adjust the directory if necessary

# Example Usage for .srt
srt_path = '/content/Incendies.srt'  # Replace this with your .srt file path
predicted_genre_ratings_srt = train_and_predict_genre(srt_path, dataframes, use_word2vec=True)

# Print the ratings for each genre from .srt
print("Multiple Genre Ratings from .srt (out of 100):")
for genre, rating in predicted_genre_ratings_srt.items():
    print(f'{genre}: {rating}%')