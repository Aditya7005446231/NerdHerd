from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Load the classification pipeline with the specified model
app = Flask(__name__)

sentiment_analysis = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

@app.route('/')
def home():
    return render_template('index.html')

# Classify a new sentence
@app.route('/analyze',methods=['POST'])

def analyze_sentiment():
    data = request.get_json()
    text = data.text('text')
    result = sentiment_analysis(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
