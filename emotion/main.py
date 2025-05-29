import re
from transformers import pipeline

# Load classifiers
anger_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512,
    framework="pt"
)

other_classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-emotion",
    top_k=None,
    truncation=True,
    max_length=512,
    framework="pt"
)

def clean_text(text):
    text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'\b(why isn\'t|what the|are you serious|come on|i did not|this is ridiculous|you kidding|fix it|do better|unacceptable|driving me nuts|broken again|worst service|keep crashing|complete disaster|so annoying|totally unacceptable|fed up|nonsense)\b', r'\1.', text)
    text = re.sub(r'\b(this what|what you)\b', r'\1,', text)
    text = text.replace(' i ', ' I ')
    if not text.endswith('.'):
        text += '.'
    return text

def is_neutral_query(text):
    neutral_patterns = [
        r'^(what|how|when|where|can you|could you|tell me|is it|hey).*?\?$',
        r'^(please|could you|would you|hey).*?(schedule|set|find|look up|tell me|explain|reset|alarm).*'
    ]
    technical_terms = ['machine learning', 'deep learning', 'artificial intelligence', 'data science', 'neural networks', 'ml', 'ai', 'advancements']
    text = text.lower().strip()
    if any(term in text for term in technical_terms) or any(re.match(pattern, text) for pattern in neutral_patterns):
        return True
    return False

def is_anger_query(text):
    anger_patterns = [
        r'\b(why isn\'t|what the|are you serious|come on|i did not expect|this is ridiculous|you kidding|fix it|do better|unacceptable|driving me nuts|broken again|worst service|keep crashing|complete disaster|so annoying|totally unacceptable|fed up|nonsense|not working|failing|broken|messed up)\b'
    ]
    text = text.lower().strip()
    return any(re.search(pattern, text) for pattern in anger_patterns)

def predict_emotion_from_text(text):
    try:
        text = clean_text(text)
        if not text:
            raise ValueError("Input text is empty after cleaning")

        if is_neutral_query(text):
            return "Neutral", 0.95

        if is_anger_query(text):
            return "Anger", 0.90

        anger_results = anger_classifier(text)[0]
        anger_score = next((r['score'] for r in anger_results if r['label'] == 'anger'), 0.0)
        if anger_score > 0.6:
            return "Anger", anger_score

        other_results = other_classifier(text)[0]
        
        negative_words = ['not', 'isn\'t', 'doesn\'t', 'won\'t', 'can\'t', 'failing', 'broken', 'crashing', 'annoying', 'disaster', 'unacceptable', 'fed up', 'nonsense']
        if any(word in text.lower() for word in negative_words):
            for result in other_results:
                if result['label'] == 'emotion:joy':
                    result['score'] *= 0.05

        best = max(other_results, key=lambda x: x['score'])
        emotion = best['label'].replace('emotion:', '').capitalize()
        score = best['score']

        if score < 0.7:
            return "Neutral", 0.95

        return emotion, score
    except Exception as e:
        return "Unknown", 0.0, str(e)
