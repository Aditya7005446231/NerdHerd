from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import random
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the sentiment analysis model
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Emotion Keywords (unchanged)
emotion_keywords = {
    'sadness': [
        "lonely", "disappointed", "heartbroken", "upset", 
        "depressed", "hopeless", "grief", "sorrow", "mourning", 
        "gloomy", "worthless", "rejected", "despair"
    ],
    'joy': [
        "excited", "thrilled", "delighted", "happy", 
        "cheerful", "content", "satisfied", "optimistic", 
        "elated", "euphoric", "proud", "grateful"
    ],
    'anger': [
        "frustrated", "annoyed", "irritated", "mad", 
        "resentful", "hostile", "outraged", "enraged", 
        "agitated", "bitter", "vindictive", "jealous"
    ],
    'fear': [
        "worried", "anxious", "nervous", "scared", 
        "panicked", "terrified", "frightened", "uneasy", 
        "paranoid", "insecure", "overwhelmed", "afraid"
    ],
    'stress': [
        "exhausted", "burned out", "tense", "pressured", 
        "overwhelmed", "frazzled", "fatigued", "stressed", 
        "restless", "uneasy", "jittery", "tired"
    ],
    'confusion': [
        "lost", "confused", "uncertain", "unsure", 
        "indecisive", "perplexed", "bewildered", "puzzled", 
        "dazed", "disoriented", "unsettled", "foggy"
    ],
    'guilt': [
        "ashamed", "guilty", "remorseful", "regretful", 
        "blameworthy", "humiliated", "disgraced", "embarrassed", 
        "self-loathing", "apologetic", "responsible", "accountable"
    ],
    'hopelessness': [
        "hopeless", "desperate", "powerless", "defeated", 
        "crushed", "discouraged", "pessimistic", "despondent", 
        "helpless", "trapped", "in despair", "forsaken"
    ]
}

# Emotion Messages
emotion_messages = {
    'sadness': {
        "lonely": "It seems like you're feeling lonely. Connecting with someone could help lift your spirits.",
        "disappointed": "I'm sorry you're feeling disappointed. Sometimes things don’t turn out as we hope, but there's always another chance.",
        "heartbroken": "It looks like you're feeling heartbroken. Surround yourself with people who care about you, and take things one step at a time.",
        "upset": "You're feeling upset right now. Take some time to relax and gather your thoughts."
    },
    'joy': {
        "excited": "You seem really excited! Keep that enthusiasm going and enjoy the moment.",
        "thrilled": "You’re thrilled, and that's amazing! Celebrate this joy and share it with others.",
        "happy": "You're radiating happiness! Keep spreading that positive energy.",
        "content": "You’re feeling content, which is wonderful. Sometimes it's the little things that bring the most peace."
    },
    'anger': {
        "frustrated": "You seem frustrated. Take a deep breath and give yourself a moment to calm down.",
        "annoyed": "You're feeling annoyed right now. Stepping back might help clear your mind.",
        "mad": "It looks like you're feeling mad. Talking about it could help release some tension."
    },
    'fear': {
        "worried": "It seems like you're feeling worried. Talk to someone you trust to help ease your mind.",
        "anxious": "You're feeling anxious. Remember, it's okay to take things one step at a time.",
        "scared": "You seem scared right now. Surround yourself with things that bring you comfort."
    },
    'stress': {
        "exhausted": "You're feeling exhausted. Take a break and prioritize your well-being.",
        "burned out": "You seem burned out. Rest and relaxation could help you regain your energy.",
        "tense": "It looks like you're feeling tense. Finding some time to unwind could be helpful."
    },
    'confusion': {
        "lost": "It seems like you're feeling lost. Take it slow and ask for guidance if needed.",
        "confused": "You’re confused right now. Breaking things down into smaller steps might help.",
        "uncertain": "You seem uncertain. Taking time to gather more information could provide clarity."
    },
    'guilt': {
        "ashamed": "You're feeling ashamed. Remember, no one is perfect, and it's okay to make mistakes.",
        "guilty": "It seems like you're feeling guilty. Forgiving yourself is the first step to moving forward.",
        "regretful": "You seem regretful. Reflecting on what happened and learning from it could help you heal."
    },
    'hopelessness': {
        "hopeless": "You're feeling hopeless. Don’t hesitate to reach out for support; there’s always a way forward.",
        "desperate": "It seems like you're feeling desperate. Sometimes sharing your thoughts with someone can lighten the load.",
        "defeated": "You seem defeated right now. Take a moment to breathe and try to regain some hope."
    }
}

# Function to detect emotion and provide response
def generate_emotion_response(input_text):
    input_text = input_text.lower()  # Convert text to lowercase to handle case insensitivity
    # Tokenize the input text into words (ignoring punctuation)
    words = re.findall(r'\b\w+\b', input_text)

    # Iterate over each emotion category
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in words:  # Partial match with individual words
                # Get specific message for detected keyword
                if keyword in emotion_messages[emotion]:
                    return emotion_messages[emotion][keyword]
    
    # Default response if no keyword is found
    return "I'm here to listen whenever you're ready to share more."

# Example Usage
input_text = "I'm feeling very lonely and hopeless today."
response = generate_emotion_response(input_text)
print(response)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

# Define a route for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the POST request
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get the prediction from the model
        prediction = classifier(text)[0]
        
        emotion = prediction.get('label')
        score = prediction.get('score')
        
        # Ensure emotion exists in the predefined messages
        if emotion in emotion_messages:
            message = random.choice(list(emotion_messages[emotion].values()))
        else:
            message = "It's okay to not feel okay."
        
        if score > 0.8 and emotion == 'sadness':
            message += " It's important to reach out to someone if you're feeling this way."
        
        # Return the prediction as a JSON response
        return jsonify({
            'emotion': emotion,
            'score': round(score, 2),
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
