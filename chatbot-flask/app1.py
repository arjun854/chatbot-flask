import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

stemmer = LancasterStemmer()

def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]
    return ' '.join(stemmed_words)

# Initialize the LabelEncoder and TfidfVectorizer
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

# Load the dataset
data = pd.read_csv('BankFAQs.csv')
questions = data['Question'].values

# Clean the questions
X = [cleanup(question) for question in questions]

# Fit the TFIDF vectorizer and label encoder
tfv.fit(X)
le.fit(data['Class'])

X = tfv.transform(X)
y = le.transform(data['Class'])

# Split the dataset into training and testing sets
trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

# Train the model using Support Vector Classifier
model = SVC(kernel='linear')
model.fit(trainx, trainy)

print("SVC model trained with accuracy:", model.score(testx, testy))

def get_max5(arr):
    return np.argsort(arr)[-5:][::-1]

# Route to display the chatbot interface
@app.route('/')
def index():
    # Send the first 5 questions when the page loads
    questionset = data.iloc[0:5]
    questions_to_display = list(questionset['Question'])
    return render_template('index.html', questions=questions_to_display)

# Route to handle the chatbot's interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input'].strip().lower()
    current_index = int(request.form['current_index'])

    # Handle "QUIT" input
    if user_input == 'quit':
        return jsonify({'response': 'It was good to be of help. Goodbye!', 'current_index': current_index, 'questions': []})

    # Handle "MORE" input
    if user_input == 'more':
        current_index += 5
        if current_index >= len(data):
            return jsonify({'response': 'No more questions available.', 'current_index': current_index, 'questions': []})
        else:
            questionset = data.iloc[current_index:current_index + 5]
            questions_to_display = list(questionset['Question'])
            return jsonify({'response': 'Here are the next set of questions:', 'current_index': current_index, 'questions': questions_to_display})

    # Handle question number input
    if user_input.isdigit():
        question_number = int(user_input)
        if 1 <= question_number <= 5:
            question_selected = data.iloc[current_index + question_number - 1]
            return jsonify({'response': question_selected['Answer'], 'current_index': current_index, 'questions': []})
        else:
            return jsonify({'response': 'Invalid selection. Please choose a number between 1 and 5.', 'current_index': current_index, 'questions': []})
    
    # If input is invalid
    return jsonify({'response': 'Invalid input. Please type a number, "MORE", or "QUIT".', 'current_index': current_index, 'questions': []})

if __name__ == '__main__':
    app.run(debug=True)
