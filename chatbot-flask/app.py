import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, request, jsonify, render_template

# Initialize the Flask app
app = Flask(__name__)

# Directory containing CSV files
CSV_DIR = "csv_files"

# Initialize tools
stemmer = LancasterStemmer()
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words="english")

# Helper function to clean sentences
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]
    return " ".join(stemmed_words)

# Load all available CSV files from the 'csv_files' directory
csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
loaded_data = None  # To store the data selected by the user
model = None  # To store the trained SVM model
questions = []  # To store the questions from the selected file

# Route for the main UI
@app.route("/")
def home():
    # List all CSV files in the directory without extensions
    csv_files = [os.path.splitext(f)[0] for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    return render_template("index.html", files=csv_files)

# Route to handle user interactions
@app.route("/chat", methods=["POST"])
def chat():
    global loaded_data, model, questions

    # Get user input and current state
    user_input = request.form["user_input"].strip().lower()
    current_index = int(request.form["current_index"])

    # If the user hasn't selected a file, prompt them to select one
    if loaded_data is None:
        try:
            file_index = int(user_input) - 1
            if 0 <= file_index < len(csv_files):
                selected_file = os.path.join(CSV_DIR, csv_files[file_index])
                loaded_data = pd.read_csv(selected_file)

                # Preprocess the data
                loaded_data['Question'] = loaded_data['Question'].apply(cleanup)
                X = tfv.fit_transform(loaded_data['Question'])
                y = le.fit_transform(loaded_data['Class'])
                trainx, testx, trainy, testy = tts(X, y, test_size=0.25, random_state=42)

                # Train the SVM model
                model = SVC(kernel="linear")
                model.fit(trainx, trainy)

                # Save the questions for later use
                questions = loaded_data["Question"].tolist()

                return jsonify({
                    "response": f"How can I help you?",
                    "questions": questions[:5],  # Show the first 5 questions
                    "current_index": 0
                })
            else:
                return jsonify({
                    "response": f"Invalid selection. Please choose a number between 1 and {len(csv_files)}.",
                    "questions": [],
                    "current_index": 0
                })
        except ValueError:
            return jsonify({
                "response": "Please type a valid number corresponding to your choice.",
                "questions": [],
                "current_index": 0
            })

    # If a file is already selected, process the input
    if user_input == "more":
        current_index += 5
        if current_index >= len(questions):
            return jsonify({
                "response": "No more questions available.",
                "questions": [],
                "current_index": current_index
            })
        return jsonify({
            "response": "Here are more questions you may be asking:",
            "questions": questions[current_index:current_index + 5],
            "current_index": current_index
        })

    if user_input.isdigit():
        question_number = int(user_input)
        if 1 <= question_number <= 5:
            question_selected = loaded_data.iloc[current_index + question_number - 1]
            answer = question_selected["Answer"]
            return jsonify({
                "response": f"Answer to your question: {answer}",
                "questions": [],
                "current_index": current_index
            })
        else:
            return jsonify({
                "response": "Invalid selection. Please choose a number between 1 and 5.",
                "questions": questions[current_index:current_index + 5],
                "current_index": current_index
            })

    if user_input == "quit":
        loaded_data = None  # Reset for a new session
        return jsonify({
            "response": "Thank you for using the chatbot. Goodbye!",
            "questions": [],
            "current_index": 0
        })

    return jsonify({
        "response": "Invalid input. Please type a number, 'MORE', or 'QUIT'.",
        "questions": questions[current_index:current_index + 5],
        "current_index": current_index
    })


if __name__ == "__main__":
    nltk.download("punkt")
    app.run(debug=True)
