from flask import Flask, request, render_template
import pickle
import scipy.sparse as sp
import numpy as np

app = Flask(__name__)

# Load trained model & vectorizers
model = pickle.load(open("best_sarcasm_model.pkl", "rb"))
tfidf_word = pickle.load(open("tfidf_word.pkl", "rb"))
tfidf_char = pickle.load(open("tfidf_char.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", output=None, user_input="")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("message", "")

    if not text:
        return render_template("index.html", output="Please enter text!", user_input="")

    # Transform input
    word_vec = tfidf_word.transform([text])
    char_vec = tfidf_char.transform([text])
    marker = sp.csr_matrix(np.zeros((1, 1)))
    final_vec = sp.hstack([word_vec, char_vec, marker])

    # Predict
    pred = model.predict(final_vec)[0]

    # Convert 0/1 to text label
    label = "Sarcastic" if pred == 1 else "Non-sarcastic"

    return render_template("index.html", output=label, user_input=text)

if __name__ == "__main__":
    app.run(debug=True)
